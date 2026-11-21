"""Supply-chain digital twin: a day-by-day simulation of a disruption.

`engine.scenario` answers *who* is exposed to a disruption. The twin answers *when* they feel
it. It does this with a deliberately small model:

* every company has an output **capacity** in [0, 1] (1 = running normally);
* a disrupted company drops to ``1 - severity`` for `duration` days, then recovers linearly
  over `recovery_days`;
* a customer's **supply** is the weighted mean of its suppliers' capacities, where each link
  is weighted by how much the graph trusts it (SUPPLIES counts more than POSSIBLE_SUPPLIER);
* every company holds `inventory_days` of input stock. A supply shortfall is covered from
  stock first. A company's capacity drops only once its stock is gone, and only to the level
  its supply can sustain.

The model is a first approximation, not a forecast: the graph does not record volumes, lead
times or substitute suppliers. What it is good for is ordering: which customers stock out
first, and how long a buffer would keep them running.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from supplygraph.risk.engine import Edge, RiskConfig

SUPPLY_RELATIONS = ("SUPPLIES", "POSSIBLE_SUPPLIER")


@dataclass(frozen=True)
class Disruption:
    lei: str
    severity: float = 1.0  # share of capacity lost
    start_day: int = 0
    duration: int = 14
    recovery_days: int = 7

    def capacity(self, day: int) -> float:
        if day < self.start_day:
            return 1.0
        floor = 1.0 - min(1.0, max(0.0, self.severity))
        t = day - self.start_day - self.duration
        if t < 0:
            return floor
        if self.recovery_days <= 0 or t >= self.recovery_days:
            return 1.0
        return floor + (1.0 - floor) * (t + 1) / (self.recovery_days + 1)


@dataclass
class TwinResult:
    days: int
    capacity: dict[str, list[float]]  # lei -> capacity per day
    inventory: dict[str, list[float]]  # lei -> days of stock left, per day
    stockout_day: dict[str, int | None] = field(default_factory=dict)

    def impacted(self, threshold: float = 0.999) -> list[dict]:
        """Companies whose capacity fell below `threshold`, earliest stockout first."""
        out = []
        for lei, series in self.capacity.items():
            low = min(series)
            if low >= threshold:
                continue
            below = [d for d, c in enumerate(series) if c < threshold]
            out.append({
                "lei": lei, "min_capacity": round(low, 4),
                "first_impact_day": below[0], "days_impacted": len(below),
                "stockout_day": self.stockout_day.get(lei),
                "lost_output_days": round(sum(1.0 - c for c in series), 3),
            })
        return sorted(out, key=lambda r: (r["first_impact_day"], r["min_capacity"]))


def supply_graph(edges: list[Edge], cfg: RiskConfig | None = None) -> dict[str, dict[str, float]]:
    """customer -> {supplier: weight}, from the risk engine's supply edges."""
    cfg = cfg or RiskConfig()
    inputs: dict[str, dict[str, float]] = defaultdict(dict)
    for e in edges:
        if e.relation in SUPPLY_RELATIONS and e.src != e.dst:
            w = cfg.edge_factor(e.relation)
            inputs[e.dst][e.src] = max(w, inputs[e.dst].get(e.src, 0.0))
    return dict(inputs)


def simulate(edges: list[Edge], disruptions: list[Disruption], days: int = 60,
             inventory_days: float | dict[str, float] = 10.0,
             cfg: RiskConfig | None = None) -> TwinResult:
    inputs = supply_graph(edges, cfg)
    leis = set(inputs) | {s for sup in inputs.values() for s in sup} | {d.lei for d in disruptions}
    shocks: dict[str, list[Disruption]] = defaultdict(list)
    for d in disruptions:
        shocks[d.lei].append(d)

    def buffer(lei: str) -> float:
        if isinstance(inventory_days, dict):
            return float(inventory_days.get(lei, 10.0))
        return float(inventory_days)

    stock = {lei: buffer(lei) for lei in leis}
    prev = {lei: 1.0 for lei in leis}  # yesterday's capacity; goods take a day to ship
    capacity: dict[str, list[float]] = {lei: [] for lei in leis}
    inventory: dict[str, list[float]] = {lei: [] for lei in leis}
    stockout: dict[str, int | None] = {lei: None for lei in leis}

    for day in range(days):
        today = {}
        for lei in leis:
            sups = inputs.get(lei)
            if sups:
                total = sum(sups.values())
                supply = sum(w * prev[s] for s, w in sups.items()) / total
            else:
                supply = 1.0
            shortfall = 1.0 - supply
            if shortfall <= 1e-12:
                # Recovering suppliers refill stock at up to 20 % of a day's need per day.
                stock[lei] = min(buffer(lei), stock[lei] + 0.2)
                cap = 1.0
            elif stock[lei] >= shortfall:
                stock[lei] -= shortfall
                cap = 1.0
            else:
                cap = supply + stock[lei]  # what's left of the stock tops up today's supply
                stock[lei] = 0.0
                if stockout[lei] is None:
                    stockout[lei] = day
            for d in shocks.get(lei, ()):
                cap = min(cap, d.capacity(day))
            today[lei] = max(0.0, min(1.0, cap))
            capacity[lei].append(today[lei])
            inventory[lei].append(round(stock[lei], 4))
        prev = today
    return TwinResult(days, capacity, inventory, stockout)
