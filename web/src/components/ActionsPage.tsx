import { useState } from "react";
import { api, apiToken, type Action } from "../api";
import { ACTION_LABELS } from "../format";
import { Loading, RefLink, useLoad } from "./common";

const ANALYST_KEY = "supplygraph.analyst";

function storedAnalyst(): string {
  try {
    return localStorage.getItem(ANALYST_KEY) ?? "";
  } catch {
    return "";
  }
}

export default function ActionsPage({ onChange }: { onChange: () => void }) {
  const [status, setStatus] = useState<string>("proposed");
  const [version, setVersion] = useState(0);
  const list = useLoad(() => api.actions(status || undefined), [status, version]);
  const config = useLoad(api.config, []);
  const writes = config.data?.writes ?? "open";
  const [analyst, setAnalyst] = useState(storedAnalyst);
  const [token, setToken] = useState(apiToken.get);

  const saveAnalyst = (name: string) => {
    setAnalyst(name);
    try {
      localStorage.setItem(ANALYST_KEY, name);
    } catch {
      /* private mode: keep it in memory only */
    }
  };

  return (
    <div className="grid" style={{ maxWidth: 1000 }}>
      <section className="card">
        <h2>Agent proposals and analyst decisions</h2>
        <p className="small muted" style={{ marginTop: 0 }}>
          The agent can only propose. Your decision is appended to the audit log
          (<span className="mono">data/actions/events.jsonl</span>) and takes effect at the next
          pipeline run: flags at <span className="mono">build</span>, sanctions decisions at <span className="mono">resolve</span>.
        </p>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <div className="seg" role="group" aria-label="Status">
            {[["proposed", "Awaiting decision"], ["approved", "Approved"], ["rejected", "Rejected"], ["", "All"]].map(([v, l]) => (
              <button key={v} aria-pressed={status === v} onClick={() => setStatus(v)}>{l}</button>
            ))}
          </div>
          {writes === "token" && (
            <label className="small">API token{" "}
              <input type="password" value={token} autoComplete="off"
                     onChange={(e) => { setToken(e.target.value); apiToken.set(e.target.value); }}
                     style={{ padding: "5px 8px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface-1)" }} />
            </label>
          )}
          <label className="small">Deciding analyst{" "}
            <input value={analyst} onChange={(e) => saveAnalyst(e.target.value)} placeholder="your name"
                   style={{ padding: "5px 8px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface-1)" }} />
          </label>
        </div>
        {writes === "off" && (
          <p className="notice small" style={{ marginBottom: 0 }}>
            This server is read-only (no API token configured), so decisions are disabled here.
            Use <span className="mono">supplygraph actions approve|reject</span>, or redeploy with a token.
          </p>
        )}
      </section>
      <Loading error={list.error} loading={list.loading && !list.data} />
      {list.data && list.data.actions.length === 0 && <p className="muted">No actions here.</p>}
      {list.data?.actions.map((a) => (
        <ActionCard key={a.action_id} action={a} analyst={analyst} canDecide={writes !== "off"}
                    onDecided={() => { setVersion((v) => v + 1); onChange(); }} />
      ))}
    </div>
  );
}

function ActionCard({ action: a, analyst, canDecide, onDecided }: { action: Action; analyst: string; canDecide: boolean; onDecided: () => void }) {
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const decide = async (approve: boolean) => {
    setError(null);
    if (!analyst.trim()) return setError("Enter the deciding analyst's name first.");
    if (!approve && !note.trim()) return setError("A rejection needs a note explaining why.");
    setBusy(true);
    try {
      await api.decide(a.action_id, approve, analyst.trim(), note.trim() || null);
      onDecided();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className="card">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <strong>{ACTION_LABELS[a.action] ?? a.action}</strong>{" "}
          <RefLink refId={`lei:${a.lei}`} label={a.lei} />
          {a.sanction_id && <> / <span className="mono">{a.sanction_id}</span></>}
        </div>
        <span className={`status ${a.status === "approved" ? "good" : a.status === "rejected" ? "neutral" : "warning"}`}>
          {a.status === "proposed" ? "Awaiting decision" : a.status[0].toUpperCase() + a.status.slice(1)}
        </span>
      </div>
      <p style={{ margin: "8px 0" }}>{a.reason}</p>
      <div className="small">
        Evidence:{" "}
        {a.evidence.map((ref, i) => (
          <span key={ref}>{i > 0 && ", "}<RefLink refId={ref} /></span>
        ))}
      </div>
      <div className="small muted" style={{ marginTop: 4 }}>
        Proposed {new Date(a.proposed_at).toLocaleString()} by {a.proposed_by}
        {a.question && <> in answer to “{a.question}”</>}
        <span className="mono"> · {a.action_id}</span>
      </div>
      {a.decided_by && (
        <div className="small" style={{ marginTop: 4 }}>
          {a.status} by {a.decided_by} at {a.decided_at && new Date(a.decided_at).toLocaleString()}
          {a.decision_note && <> — {a.decision_note}</>}
        </div>
      )}
      {a.status === "proposed" && canDecide && (
        <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Note (required to reject)"
                 aria-label="Decision note" style={{ flex: 1, minWidth: 200, padding: "6px 8px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface-1)" }} />
          <button className="btn primary" disabled={busy} onClick={() => decide(true)}>Approve</button>
          <button className="btn" disabled={busy} onClick={() => decide(false)}>Reject</button>
        </div>
      )}
      {error && <p className="error small" role="alert">{error}</p>}
    </article>
  );
}
