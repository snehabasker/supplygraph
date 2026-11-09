import { useState } from "react";
import { api, leiOf } from "../api";
import { ACTION_LABELS, fmtScore } from "../format";
import { Loading, RefLink, useLoad } from "./common";

const RELATIONS = [
  { key: "suppliers", label: "Suppliers" },
  { key: "customers", label: "Customers" },
  { key: "parents", label: "Parents" },
  { key: "subsidiaries", label: "Subsidiaries" },
] as const;

export default function CompanyPanel({ lei }: { lei: string }) {
  const company = useLoad(() => api.company(lei), [lei]);
  const news = useLoad(() => api.news(lei), [lei]);
  const [relation, setRelation] = useState<string>("suppliers");
  const rel = useLoad(() => api.relationships(lei, relation), [lei, relation]);
  const c = company.data;

  return (
    <section className="card panel" aria-label="Company details">
      <Loading error={company.error} loading={company.loading && !c} />
      {c && (
        <>
          <h2 style={{ color: "var(--text-primary)", fontSize: 16, marginBottom: 2 }}>{c.name}</h2>
          {c.other_names && <div className="small muted">{c.other_names.split(";").slice(0, 3).join(" · ")}</div>}
          <div className="small" style={{ marginTop: 6 }}>
            <span className="mono">{leiOf(c.ref)}</span> · {c.legal_city ? `${c.legal_city}, ` : ""}{c.legal_country} · {c.entity_status}
          </div>
          <div style={{ marginTop: 6 }}>
            {c.sanctions_matches.length > 0 && <span className="status critical">Sanctioned</span>}{" "}
            {c.sanctions_candidates?.some((x) => x.decision === "review") && <span className="status warning">Review candidate</span>}{" "}
            {c.watchlist && c.watchlist.length > 0 && <span className="status neutral">Watchlist ({c.watchlist.join(", ")})</span>}
          </div>

          <h3>Risk {c.risk ? fmtScore(c.risk.score) : "–"}
            {c.risk && c.risk.score > 0 && <span className="muted small"> · own {fmtScore(c.risk.own)} · inherited {fmtScore(c.risk.inherited)}</span>}
          </h3>
          {c.risk?.drivers.length === 0 && <p className="small muted">No risk signal reaches this company.</p>}
          {c.risk?.drivers.map((d, i) => (
            <div className="driver" key={i}>
              <div className="small"><strong>{d.signal.replace("_", " ").toLowerCase()}</strong> · contributes {fmtScore(d.value)}</div>
              <div className="small">{d.detail}</div>
              {d.hops > 0 && <div className="path">{d.path.join(" → ")} <span className="muted">[{d.relations.join(", ")}]</span></div>}
            </div>
          ))}

          {c.sanctions_matches.length > 0 && (
            <>
              <h3>Sanctions matches</h3>
              {c.sanctions_matches.map((m) => (
                <div key={m.sanction_ref} className="small">
                  {m.sanction_name} <span className="mono muted">{m.sanction_ref}</span>
                  <div className="muted">{m.match_method}, confidence {m.confidence}</div>
                </div>
              ))}
            </>
          )}
          {c.sanctions_candidates && c.sanctions_candidates.length > 0 && (
            <>
              <h3>Sanctions candidates</h3>
              {c.sanctions_candidates.map((m) => (
                <div key={m.sanction_ref} className="small">
                  {m.sanction_name} <span className="mono muted">{m.sanction_ref}</span>
                  <div className="muted">score {m.score} · {m.decision} ({m.method})</div>
                  {m.reasoning && <div className="evidence">{m.reasoning}</div>}
                </div>
              ))}
            </>
          )}
          {c.analyst_actions && c.analyst_actions.length > 0 && (
            <>
              <h3>Analyst actions</h3>
              {c.analyst_actions.map((a) => (
                <div key={a.action_id} className="small">
                  <a href="#/actions">{ACTION_LABELS[a.action] ?? a.action}</a> · {a.status} <span className="muted">{a.reason}</span>
                </div>
              ))}
            </>
          )}

          <h3>Links</h3>
          <div className="seg" role="group" aria-label="Relationship type">
            {RELATIONS.map((r) => (
              <button key={r.key} aria-pressed={relation === r.key} onClick={() => setRelation(r.key)}>{r.label}</button>
            ))}
          </div>
          <Loading error={rel.error} loading={rel.loading} />
          {rel.data && (
            <>
              {rel.data.items.length === 0 && <p className="small muted">None recorded in the graph (which is not the same as none existing).</p>}
              <ul style={{ paddingLeft: 18 }}>
                {rel.data.items.map((it) => (
                  <li key={it.ref + (it.filing_ref ?? "")} className="small" style={{ marginBottom: 6 }}>
                    <RefLink refId={it.ref} label={it.name} />{" "}
                    {it.link_type && <span className="muted">{it.link_type === "POSSIBLE_SUPPLIER" ? "possible (unverified)" : "asserted"}</span>}
                    {it.relationship_type && <span className="muted">{it.relationship_type.toLowerCase()}</span>}
                    {it.evidence && <div className="evidence">“{it.evidence}”</div>}
                    {it.filing_ref && <div className="muted">{it.form} {it.filing_date} · <span className="mono">{it.filing_ref}</span></div>}
                  </li>
                ))}
              </ul>
              <p className="small muted">{rel.data.note}</p>
            </>
          )}

          <h3>Disruption news</h3>
          <Loading error={news.error} loading={news.loading} />
          {news.data && news.data.articles.length === 0 && <p className="small muted">No linked articles.</p>}
          {news.data?.articles.map((a) => (
            <div key={a.ref} className="small" style={{ marginBottom: 6 }}>
              <a href={a.url} target="_blank" rel="noreferrer">{a.title || a.url}</a>
              <div className="muted">{a.domain} · {new Date(a.published_at).toLocaleString()} · {a.event_types.join(", ")}</div>
            </div>
          ))}
          {news.data && news.data.articles.length > 0 && (
            <p className="small muted">A mention is a signal, not proof the company was affected.</p>
          )}
        </>
      )}
    </section>
  );
}
