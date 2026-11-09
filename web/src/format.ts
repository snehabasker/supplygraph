// Pure helpers (unit-tested in format.test.ts).

/** Sequential blue ramp (reference palette steps 250 -> 700): magnitude, light -> dark. */
export const RISK_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"];

/** Upper bounds of the ramp steps; matches the dashboard histogram buckets. */
export const RISK_BOUNDS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0];

/** Fill for a risk score: zero risk recedes to the surface token, the rest step up the ramp. */
export function riskColor(score: number, zero = "var(--risk-0)"): string {
  if (!(score > 0)) return zero;
  const i = RISK_BOUNDS.findIndex((b) => score <= b);
  return RISK_RAMP[i === -1 ? RISK_RAMP.length - 1 : i];
}

export function fmtScore(score: number | null | undefined): string {
  if (score == null) return "–";
  if (score === 0) return "0";
  return score < 0.01 ? "<0.01" : score.toFixed(2);
}

export const REF_PATTERN = /\[([^\]]+)\]/g;

/** Split an agent answer into text and cited refs ("[lei:A, news:B]" -> two refs). */
export function splitCitations(text: string): (string | { ref: string })[] {
  const out: (string | { ref: string })[] = [];
  let last = 0;
  for (const m of text.matchAll(REF_PATTERN)) {
    const refs = m[1].split(",").map((s) => s.trim()).filter((s) => /^(lei|sanction|filing|news):\S+$/.test(s));
    if (refs.length === 0) continue;
    out.push(text.slice(last, m.index));
    refs.forEach((ref) => out.push({ ref }));
    last = (m.index ?? 0) + m[0].length;
  }
  out.push(text.slice(last));
  return out.filter((p) => p !== "");
}

export const ACTION_LABELS: Record<string, string> = {
  flag_supplier: "Flag supplier",
  request_review: "Request review",
  confirm_match: "Confirm sanctions match",
  dismiss_match: "Dismiss sanctions match",
};

export const EDGE_LABELS: Record<string, string> = {
  OWNED_BY: "Owned by (GLEIF)",
  SUPPLIES: "Supplies (LLM, asserted)",
  POSSIBLE_SUPPLIER: "Possible supplier (rules, unverified)",
  PARENT_OF: "Parent of",
};
