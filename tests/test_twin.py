"""Digital-twin simulation: timing, buffers, recovery."""

from __future__ import annotations

import pytest

from supplygraph.risk.engine import Edge
from supplygraph.risk.twin import Disruption, simulate


def supplies(s: str, c: str, rel: str = "SUPPLIES") -> Edge:
    return Edge(s, c, rel)


def test_disruption_capacity_profile() -> None:
    d = Disruption("S", severity=0.8, start_day=2, duration=3, recovery_days=1)
    assert [round(d.capacity(t), 2) for t in range(7)] == [1, 1, 0.2, 0.2, 0.2, 0.6, 1]


def test_no_disruption_means_no_impact() -> None:
    r = simulate([supplies("S", "C")], [], days=20)
    assert r.impacted() == []


def test_inventory_delays_stockout_and_bigger_buffer_delays_more() -> None:
    edges = [supplies("S", "C")]
    shock = [Disruption("S", 1.0, duration=30, recovery_days=0)]
    small = simulate(edges, shock, days=40, inventory_days=3)
    big = simulate(edges, shock, days=40, inventory_days=10)
    # Supplier goes down on day 0; the customer feels it from day 1 and burns 1 day of stock/day.
    assert small.stockout_day["C"] == 4
    assert big.stockout_day["C"] == 11
    assert min(small.capacity["C"]) == 0.0


def test_impact_cascades_with_increasing_delay() -> None:
    edges = [supplies("S", "A"), supplies("A", "B")]
    r = simulate(edges, [Disruption("S", 1.0, duration=30)], days=60, inventory_days=2)
    assert r.stockout_day["A"] < r.stockout_day["B"]
    order = [i["lei"] for i in r.impacted()]
    assert order.index("A") < order.index("B")


def test_second_supplier_softens_the_blow() -> None:
    edges = [supplies("S", "C"), supplies("T", "C")]
    r = simulate(edges, [Disruption("S", 1.0, duration=30)], days=40, inventory_days=0)
    assert min(r.capacity["C"]) == pytest.approx(0.5)


def test_full_recovery_after_disruption_ends() -> None:
    edges = [supplies("S", "C")]
    r = simulate(edges, [Disruption("S", 1.0, duration=5, recovery_days=2)], days=40,
                 inventory_days=1)
    assert r.capacity["C"][-1] == 1.0
    assert r.capacity["S"][-1] == 1.0


def test_ownership_links_do_not_carry_physical_supply() -> None:
    edges = [Edge("P", "C", "OWNED_BY:DIRECT")]
    r = simulate(edges, [Disruption("P", 1.0)], days=20, inventory_days=0)
    assert "C" not in r.capacity
