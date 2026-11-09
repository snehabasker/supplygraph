import { api, leiOf } from "../api";
import { go } from "../App";
import { fmtScore } from "../format";
import { Loading, RefLink, useLoad } from "./common";
import RiskHistogram from "./RiskHistogram";

export default function Dashboard() {
  const summary = useLoad(api.summary, []);
  const risks = useLoad(() => api.risks(15), []);
  const queue = useLoad(api.reviewQueue, []);
  const s = summary.data;

  return (
    <div className="grid">
      <Loading error={summary.error} loading={summary.loading && !s} />
      {s && (
        <section className="grid kpis" aria-label="Key figures">
          <Kpi label="Companies in graph" value={s.companies} sub={`${s.ownership_links} ownership links`} />
          <Kpi label="At risk (score > 0)" value={s.at_risk} sub={`${s.companies - s.at_risk} with no signal`} />
          <Kpi label="Sanctioned" value={s.sanctioned} sub="identifier / confirmed matches" />
          <Kpi label="Sanctions review queue" value={s.review_queue} sub="possible matches, not findings" href="#/actions" />
          <Kpi label="Supplier links" value={s.supply_links.SUPPLIES + s.supply_links.POSSIBLE_SUPPLIER}
               sub={`${s.supply_links.SUPPLIES} asserted · ${s.supply_links.POSSIBLE_SUPPLIER} unverified`} />
          <Kpi label="Actions awaiting decision" value={s.proposed_actions} sub={`watchlist: ${s.watchlist}`} href="#/actions" />
        </section>
      )}
      <div className="grid two">
        <section className="card">
          <h2>Highest risk</h2>
          <Loading error={risks.error} loading={risks.loading} />
          {risks.data && (
            <table>
              <thead>
                <tr><th>Company</th><th className="num">Risk</th><th className="num">Own</th><th>Main driver</th></tr>
              </thead>
              <tbody>
                {risks.data.companies.map((r) => (
                  <tr key={r.ref} className="clickable" onClick={() => go(`company/${leiOf(r.ref)}`)}>
                    <td><RefLink refId={r.ref} label={r.name} /> <span className="muted small">{r.country}</span></td>
                    <td className="num">{fmtScore(r.risk_score)}</td>
                    <td className="num">{fmtScore(r.own_risk)}</td>
                    <td className="small">
                      {r.top_driver?.signal.replace("_", " ").toLowerCase()}
                      {r.top_driver && r.top_driver.hops > 0 && <span className="muted"> · {r.top_driver.hops} hop(s) via {r.top_driver.relations.join(", ")}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
        <div className="grid" style={{ alignContent: "start" }}>
          {s && (
            <section className="card">
              <h2>Risk distribution</h2>
              <RiskHistogram summary={s} />
            </section>
          )}
          <section className="card">
            <h2>Sanctions review queue</h2>
            <Loading error={queue.error} loading={queue.loading} />
            {queue.data && queue.data.candidates.length === 0 && <p className="muted">Empty.</p>}
            {queue.data && queue.data.candidates.length > 0 && (
              <table>
                <thead><tr><th>Company</th><th>Sanctions record</th><th className="num">Score</th></tr></thead>
                <tbody>
                  {queue.data.candidates.map((c) => (
                    <tr key={c.company_ref + c.sanction_ref}>
                      <td><RefLink refId={c.company_ref} label={c.company_name} /></td>
                      <td>{c.sanction_name} <div className="mono muted">{c.sanction_ref}</div></td>
                      <td className="num">{c.score.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <p className="small muted">Ask the agent to confirm or dismiss a candidate; approve the proposal under Actions.</p>
          </section>
        </div>
      </div>
      {s?.scored_at && <p className="small muted">Risk scored {new Date(s.scored_at).toLocaleString()}.</p>}
    </div>
  );
}

function Kpi({ label, value, sub, href }: { label: string; value: number; sub?: string; href?: string }) {
  const body = (
    <>
      <div className="label">{label}</div>
      <div className="value">{value.toLocaleString()}</div>
      {sub && <div className="sub">{sub}</div>}
    </>
  );
  return <div className="card kpi">{href ? <a href={href} style={{ color: "inherit", textDecoration: "none" }}>{body}</a> : body}</div>;
}
