import { useCallback, useEffect, useState } from "react";
import { api, leiOf, type GraphPayload, type Scenario, type SearchMatch } from "../api";
import { go } from "../App";
import { fmtScore } from "../format";
import CompanyPanel from "./CompanyPanel";
import { Loading, useLoad } from "./common";
import GraphView from "./GraphView";

const DEFAULT_LEI = "549300KB6NK5SBD14S87"; // TSMC: the most connected supplier in the data

export default function Explorer({ lei }: { lei?: string }) {
  const center = lei ?? DEFAULT_LEI;
  const [depth, setDepth] = useState(1);
  const [mode, setMode] = useState<"neighborhood" | "scenario">("neighborhood");
  const [severity, setSeverity] = useState(1);
  const [selected, setSelected] = useState(center);
  useEffect(() => {
    setSelected(center);
    setMode("neighborhood");
  }, [center]);

  const graph = useLoad<GraphPayload | Scenario>(
    () => (mode === "scenario" ? api.scenario(center, severity) : api.graph(center, depth)),
    [center, depth, mode, severity],
  );
  const onSelect = useCallback((id: string) => setSelected(id), []);
  const scenario = mode === "scenario" && graph.data && "impacted" in graph.data ? (graph.data as Scenario) : null;

  return (
    <div className="explorer">
      <section className="card graph-card" aria-label="Graph">
        <div className="graph-toolbar">
          <Search />
          <div className="seg" role="group" aria-label="View">
            <button aria-pressed={mode === "neighborhood"} onClick={() => setMode("neighborhood")}>Neighbourhood</button>
            <button aria-pressed={mode === "scenario"} onClick={() => setMode("scenario")}>Disruption scenario</button>
          </div>
          {mode === "neighborhood" ? (
            <label className="small">Depth{" "}
              <select value={depth} onChange={(e) => setDepth(Number(e.target.value))}>
                {[1, 2, 3].map((d) => <option key={d} value={d}>{d} hop{d > 1 ? "s" : ""}</option>)}
              </select>
            </label>
          ) : (
            <label className="small">Severity{" "}
              <select value={severity} onChange={(e) => setSeverity(Number(e.target.value))}>
                {[1, 0.75, 0.5, 0.25].map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
          )}
          {graph.data && (
            <span className="small muted">
              {graph.data.nodes.length} companies · {graph.data.edges.length} links
              {graph.data.truncated && " · truncated at 150 companies"}
            </span>
          )}
        </div>
        <Loading error={graph.error} loading={graph.loading && !graph.data} />
        {graph.data && (
          <GraphView nodes={graph.data.nodes} edges={graph.data.edges} center={leiOf(graph.data.center)} mode={mode} onSelect={onSelect} />
        )}
        {scenario && (
          <div style={{ padding: "8px 12px", borderTop: "1px solid var(--border)", maxHeight: 180, overflow: "auto" }}>
            {scenario.impacted.length === 0 ? (
              <p className="small muted">No company's risk rises: disruption travels only along supplier links and from a subsidiary to its parent.</p>
            ) : (
              <table className="small">
                <thead><tr><th>Impacted company</th><th className="num">Risk increase</th><th>Path</th></tr></thead>
                <tbody>
                  {scenario.impacted.map((r) => (
                    <tr key={r.ref} className="clickable" onClick={() => setSelected(leiOf(r.ref))}>
                      <td>{r.name}</td>
                      <td className="num">+{fmtScore(r.increase)}</td>
                      <td className="muted">{r.path.join(" → ")} [{r.relations.join(", ")}]</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </section>
      <div style={{ display: "flex", flexDirection: "column", gap: 8, minHeight: 0 }}>
        {selected !== center && (
          <button className="btn" onClick={() => go(`company/${selected}`)}>Centre the graph on this company</button>
        )}
        <CompanyPanel lei={selected} />
      </div>
    </div>
  );
}

function Search() {
  const [text, setText] = useState("");
  const [matches, setMatches] = useState<SearchMatch[]>([]);
  useEffect(() => {
    if (text.trim().length < 2) {
      setMatches([]);
      return;
    }
    const t = setTimeout(() => {
      api.search(text).then((r) => setMatches(r.matches)).catch(() => setMatches([]));
    }, 200);
    return () => clearTimeout(t);
  }, [text]);

  return (
    <div className="search">
      <input type="search" placeholder="Search company, short name or LEI (e.g. TSMC)" value={text}
             aria-label="Search companies" onChange={(e) => setText(e.target.value)}
             onKeyDown={(e) => {
               if (e.key === "Enter" && matches[0]) {
                 go(`company/${leiOf(matches[0].ref)}`);
                 setText("");
               }
             }} />
      {matches.length > 0 && (
        <ul role="listbox">
          {matches.map((m) => (
            <li key={m.ref}>
              <button onClick={() => { go(`company/${leiOf(m.ref)}`); setText(""); }}>
                {m.name} <span className="muted small">{m.country} · risk {fmtScore(m.risk_score)}</span>
                {m.other_names && <div className="muted small">{m.other_names.split(";")[0]}</div>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
