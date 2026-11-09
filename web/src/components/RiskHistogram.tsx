import { useState } from "react";
import type { Summary } from "../api";

// One series (companies per risk band): single color, no legend; the title names it.
// Bars are thin with a rounded data-end anchored to the baseline, a 2px surface gap,
// a per-bar hover tooltip, and a table view for screen readers.
export default function RiskHistogram({ summary }: { summary: Summary }) {
  const bins = summary.risk_histogram;
  const [hover, setHover] = useState<number | null>(null);
  const W = 480, H = 200, padL = 36, padB = 34, padT = 22;
  const max = Math.max(1, ...bins.map((b) => b.companies));
  const ticks = niceTicks(max);
  const top = ticks[ticks.length - 1];
  const bw = (W - padL) / bins.length;
  const y = (v: number) => padT + (H - padT - padB) * (1 - v / top);

  return (
    <figure style={{ margin: 0, position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-labelledby="hist-title">
        <title id="hist-title">Companies by risk score band (non-zero scores)</title>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} x2={W} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth={1} />
            <text x={padL - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-muted)">{t}</text>
          </g>
        ))}
        {bins.map((b, i) => {
          const x = padL + i * bw + 1; // 2px gap between neighbours
          const h = y(0) - y(b.companies);
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              {/* Hit target: the whole column, bigger than the mark. */}
              <rect x={padL + i * bw} y={padT} width={bw} height={H - padT - padB} fill="transparent" />
              {b.companies > 0 && (
                <path d={roundedTop(x + bw * 0.2, y(b.companies), bw * 0.6 - 2, h, 4)}
                      fill="var(--series-1)" opacity={hover === null || hover === i ? 1 : 0.55} />
              )}
              <text x={padL + i * bw + bw / 2} y={H - padB + 16} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
                ≤{b.to}
              </text>
            </g>
          );
        })}
        <line x1={padL} x2={W} y1={y(0)} y2={y(0)} stroke="var(--text-muted)" strokeWidth={1} />
        <text x={padL + (W - padL) / 2} y={H - 2} textAnchor="middle" fontSize={11} fill="var(--text-muted)">risk score</text>
        <text x={4} y={padT - 2} fontSize={11} fill="var(--text-muted)">companies</text>
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: `${((padL + hover * bw + bw / 2) / W) * 100}%`, top: 0, transform: "translateX(-50%)" }}>
          <strong>{bins[hover].companies}</strong> companies with risk in ({bins[hover].from}, {bins[hover].to}]
        </div>
      )}
      <figcaption className="small muted">
        {summary.zero_risk} companies have no risk signal and are not shown.
      </figcaption>
      <table className="sr-only" style={{ position: "absolute", left: -9999 }}>
        <caption>Companies by risk band</caption>
        <tbody>
          {bins.map((b, i) => (
            <tr key={i}><th>({b.from}, {b.to}]</th><td>{b.companies}</td></tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

function niceTicks(max: number): number[] {
  const step = max <= 5 ? 1 : max <= 10 ? 2 : max <= 25 ? 5 : max <= 50 ? 10 : Math.ceil(max / 50) * 10;
  const out = [];
  for (let t = 0; t <= max + step - 1; t += step) out.push(t);
  if (out[out.length - 1] < max) out.push(out[out.length - 1] + step);
  return out;
}

/** A bar with only its top corners rounded (data end), flat on the baseline. */
function roundedTop(x: number, y: number, w: number, h: number, r: number): string {
  const rr = Math.min(r, w / 2, h);
  return `M${x},${y + h} V${y + rr} Q${x},${y} ${x + rr},${y} H${x + w - rr} Q${x + w},${y} ${x + w},${y + rr} V${y + h} Z`;
}
