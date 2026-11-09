import { useState } from "react";
import { ApiError, api, type AskResult } from "../api";
import { ACTION_LABELS, splitCitations } from "../format";
import { RefLink } from "./common";

const EXAMPLES = [
  "Why is Rosneft Deutschland GmbH's risk score so high?",
  "Who supplies NVIDIA according to its annual report, and how reliable are those links?",
  "If TSMC were disrupted, which companies would be hit hardest?",
];

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [allowActions, setAllowActions] = useState(true);
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const ask = async () => {
    if (!question.trim()) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.ask(question.trim(), allowActions));
    } catch (e) {
      setError(e instanceof ApiError && e.status === 503
        ? `${e.message} The rest of the explorer works without it.`
        : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid" style={{ maxWidth: 900 }}>
      <section className="card">
        <h2>Ask the graph</h2>
        <p className="small muted" style={{ marginTop: 0 }}>
          Claude answers from the graph's read-only tools and cites every fact. Actions it takes are
          proposals for you to approve under Actions. Needs Claude API credentials with credit.
        </p>
        <textarea value={question} onChange={(e) => setQuestion(e.target.value)} aria-label="Question"
                  placeholder={EXAMPLES[0]}
                  onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask(); }} />
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
          <button className="btn primary" onClick={ask} disabled={busy || !question.trim()}>
            {busy ? "Thinking…" : "Ask"}
          </button>
          <label className="small"><input type="checkbox" checked={allowActions} onChange={(e) => setAllowActions(e.target.checked)} /> allow action proposals</label>
          <span className="small muted">Try:</span>
          {EXAMPLES.slice(1).map((ex) => (
            <button key={ex} className="btn small" onClick={() => setQuestion(ex)}>{ex.split(" ").slice(0, 4).join(" ")}…</button>
          ))}
        </div>
      </section>
      {error && <p className="notice error" role="alert">{error}</p>}
      {result && (
        <section className="card" aria-live="polite">
          {result.error && <p className="error">Session ended with an error: {result.error}</p>}
          <div className="answer">
            {splitCitations(result.answer).map((p, i) =>
              typeof p === "string" ? <span key={i}>{p}</span> : <sup key={i}>[<RefLink refId={p.ref} />]</sup>,
            )}
          </div>
          {result.ungrounded.length > 0 && (
            <p className="status critical">Unverified: cites {result.ungrounded.join(", ")}, which no tool returned.</p>
          )}
          {result.proposals.map((p) => (
            <p key={p.action_id} className="notice small">
              {ACTION_LABELS[p.action] ?? p.action} {p.lei}{p.sanction_id && ` / ${p.sanction_id}`}:{" "}
              {p.created ? "proposed" : `already ${p.status}`} — <a href="#/actions">review under Actions</a> ({p.action_id})
            </p>
          ))}
          <p className="small muted">
            {result.tool_calls.length} tool calls ({result.tool_calls.map((t) => t.name).join(", ")}) · session {result.session_id}
          </p>
        </section>
      )}
    </div>
  );
}
