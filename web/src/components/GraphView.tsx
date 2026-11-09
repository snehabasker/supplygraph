import cytoscape from "cytoscape";
import { useEffect, useRef, useState } from "react";
import type { GEdge, GNode } from "../api";
import { EDGE_LABELS, fmtScore, riskColor } from "../format";

interface Props {
  nodes: GNode[];
  edges: GEdge[];
  center: string;
  mode: "neighborhood" | "scenario";
  onSelect: (lei: string) => void;
}

/** Reads a CSS custom property, so the graph follows the light/dark theme tokens. */
function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

export default function GraphView({ nodes, edges, center, mode, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; node: GNode } | null>(null);

  useEffect(() => {
    if (!box.current) return;
    // Node fill encodes magnitude (risk, or the scenario's risk increase) on one blue ramp.
    // Status (sanctioned / review) is a ring in a reserved status color, and is also named
    // in the tooltip, the panel and the legend - never color alone.
    const zero = token("--risk-0");
    const elements: cytoscape.ElementDefinition[] = [
      ...nodes.map((n) => ({
        data: {
          id: n.id,
          label: shortName(n.name),
          fill: riskColor(mode === "scenario" ? (n.delta ?? 0) : n.risk, zero),
          ring: n.sanctioned ? token("--critical") : n.review ? token("--warning") : token("--surface-1"),
          ringWidth: n.sanctioned || n.review ? 4 : 2,
          size: n.id === center ? 46 : 26,
          level: mode === "scenario" ? 0 : 10 - (n.hops ?? 0),
        },
        classes: n.id === center ? "center" : "",
      })),
      ...edges.map((e) => ({
        data: { id: e.id, source: e.source, target: e.target, type: e.type },
        classes: e.type,
      })),
    ];
    const cy = cytoscape({
      container: box.current,
      elements,
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 0.3,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(fill)",
            "border-color": "data(ring)",
            "border-width": "data(ringWidth)",
            width: "data(size)",
            height: "data(size)",
            label: "data(label)",
            color: token("--text-primary"),
            "font-size": 10,
            "text-valign": "bottom",
            "text-margin-y": 4,
            "text-background-color": token("--surface-1"),
            "text-background-opacity": 0.8,
            "text-background-padding": "1px",
            "text-max-width": "110px",
            "text-wrap": "ellipsis",
          },
        },
        { selector: "node.center", style: { "font-size": 12, "font-weight": "bold" } },
        { selector: "node:selected", style: { "overlay-color": token("--accent"), "overlay-opacity": 0.15 } },
        {
          selector: "edge",
          style: {
            width: 2,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.9,
            "line-color": token("--ownership"),
            "target-arrow-color": token("--ownership"),
          },
        },
        {
          selector: "edge.SUPPLIES, edge.POSSIBLE_SUPPLIER",
          style: { "line-color": token("--supply"), "target-arrow-color": token("--supply") },
        },
        // Unverified links are dashed: link type never depends on color alone.
        { selector: "edge.POSSIBLE_SUPPLIER", style: { "line-style": "dashed" } },
        { selector: "edge.PARENT_OF", style: { "line-style": "dotted" } },
      ],
      // Deterministic layouts: rings by hop distance around the company, or the disruption
      // flowing outward from the shocked company.
      layout: (mode === "scenario"
        ? { name: "breadthfirst", directed: true, roots: [center], spacingFactor: 1.3, padding: 40 }
        : { name: "concentric", concentric: (n: cytoscape.NodeSingular) => n.data("level"),
            levelWidth: () => 1, minNodeSpacing: 36, padding: 40 }) as cytoscape.LayoutOptions,
    });
    // Small graphs would otherwise be fitted at a huge zoom.
    if (cy.zoom() > 1.3) {
      cy.zoom(1.3);
      cy.center();
    }
    const byId = new Map(nodes.map((n) => [n.id, n]));
    cy.on("tap", "node", (ev) => onSelect(ev.target.id()));
    cy.on("mouseover", "node", (ev) => {
      const p = ev.target.renderedPosition();
      const node = byId.get(ev.target.id());
      if (node) setTip({ x: p.x, y: p.y, node });
    });
    cy.on("mouseout", "node", () => setTip(null));
    cy.on("viewport", () => setTip(null));
    return () => cy.destroy();
  }, [nodes, edges, center, mode, onSelect]);

  return (
    <>
      <div className="graph" ref={box} role="img"
           aria-label={`Graph of ${nodes.length} companies and ${edges.length} links; the company list is in the side panel`}>
        {tip && (
          <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }}>
            <strong>{tip.node.name}</strong> <span className="muted">{tip.node.country}</span>
            <div>Risk {fmtScore(tip.node.risk)}{mode === "scenario" && <> · +{fmtScore(tip.node.delta ?? 0)} in scenario</>}</div>
            {tip.node.sanctioned && <div className="status critical">Sanctioned</div>}
            {tip.node.review && <div className="status warning">Sanctions review candidate</div>}
          </div>
        )}
      </div>
      <div className="legend" aria-label="Legend">
        <span>
          <svg className="swatch" width="64" height="10" aria-hidden>
            {[0.02, 0.07, 0.2, 0.4, 0.6, 0.9].map((v, i) => <rect key={v} x={i * 10.5} width="10" height="10" rx="2" fill={riskColor(v)} />)}
          </svg>
          {mode === "scenario" ? "risk increase (low → high)" : "risk score (low → high)"}
        </span>
        <span><svg className="swatch" width="12" height="12" aria-hidden><circle cx="6" cy="6" r="4" fill="none" stroke="var(--critical)" strokeWidth="3" /></svg>sanctioned</span>
        <span><svg className="swatch" width="12" height="12" aria-hidden><circle cx="6" cy="6" r="4" fill="none" stroke="var(--warning)" strokeWidth="3" /></svg>review candidate</span>
        {mode === "neighborhood" ? (
          <>
            <EdgeKey color="var(--ownership)" label={EDGE_LABELS.OWNED_BY} />
            <EdgeKey color="var(--supply)" label={EDGE_LABELS.SUPPLIES} />
            <EdgeKey color="var(--supply)" dash label={EDGE_LABELS.POSSIBLE_SUPPLIER} />
          </>
        ) : (
          <span className="muted">arrows show the direction the disruption travels</span>
        )}
      </div>
    </>
  );
}

function EdgeKey({ color, label, dash }: { color: string; label: string; dash?: boolean }) {
  return (
    <span>
      <svg className="swatch" width="26" height="8" aria-hidden>
        <line x1="0" x2="26" y1="4" y2="4" stroke={color} strokeWidth="2" strokeDasharray={dash ? "5 3" : undefined} />
      </svg>
      {label}
    </span>
  );
}

function shortName(name: string): string {
  return name.length > 28 ? `${name.slice(0, 27)}…` : name;
}
