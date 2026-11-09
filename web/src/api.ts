// Typed client for the SupplyGraph API (src/supplygraph/api/app.py).

export interface Summary {
  companies: number;
  at_risk: number;
  zero_risk: number;
  sanctioned: number;
  review_queue: number;
  watchlist: number;
  ownership_links: number;
  supply_links: { SUPPLIES: number; POSSIBLE_SUPPLIER: number };
  news_links: number;
  proposed_actions: number;
  risk_histogram: { from: number; to: number; companies: number }[];
  scored_at: string | null;
}

export interface Driver {
  signal: string;
  detail: string;
  value: number;
  hops: number;
  path: string[];
  relations: string[];
  source_ref: string;
  sanction_ref?: string;
}

export interface RiskRow {
  ref: string;
  name: string;
  country: string;
  risk_score: number;
  own_risk: number;
  inherited_risk: number;
  top_driver: Driver | null;
}

export interface SearchMatch {
  ref: string;
  name: string;
  other_names: string | null;
  country: string;
  risk_score: number | null;
}

export interface Company {
  ref: string;
  name: string;
  other_names: string | null;
  registered_as: string | null;
  legal_country: string;
  legal_city: string | null;
  entity_status: string;
  watchlist?: string[];
  risk?: { score: number; own: number; inherited: number; drivers: Driver[] };
  sanctions_matches: { sanction_ref: string; sanction_name: string; match_method: string; confidence: number }[];
  sanctions_candidates?: { sanction_ref: string; sanction_name: string; score: number; decision: string; method: string; reasoning: string | null }[];
  analyst_actions?: { action_id: string; action: string; status: string; reason: string; proposed_by: string; decided_by: string | null }[];
}

export interface RelItem {
  ref: string;
  name: string;
  relationship_type?: string;
  country?: string;
  link_type?: string;
  supplies?: string | null;
  evidence?: string | null;
  filing_ref?: string;
  form?: string;
  filing_date?: string;
}

export interface Relationships { relation: string; total: number; items: RelItem[]; note: string }

export interface Article {
  ref: string;
  published_at: string;
  title: string;
  url: string;
  domain: string;
  event_types: string[];
}

export interface GNode {
  id: string;
  name: string;
  country: string;
  risk: number;
  own_risk: number;
  watchlist?: boolean;
  sanctioned?: boolean;
  review?: boolean;
  hops?: number;
  delta?: number;
  shocked?: boolean;
}

export interface GEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  relationship_type?: string;
  relation?: string;
  evidence?: string | null;
  accession?: string;
}

export interface GraphPayload { center: string; nodes: GNode[]; edges: GEdge[]; truncated?: boolean }

export interface Scenario extends GraphPayload {
  severity: number;
  impacted: { ref: string; name: string; increase: number; risk_before: number; risk_after: number; path: string[]; relations: string[] }[];
}

export interface Action {
  action_id: string;
  action: string;
  lei: string;
  sanction_id: string | null;
  reason: string;
  evidence: string[];
  status: "proposed" | "approved" | "rejected";
  proposed_by: string;
  proposed_at: string;
  session_id: string | null;
  question: string | null;
  decided_by: string | null;
  decided_at: string | null;
  decision_note: string | null;
}

export interface AskResult {
  session_id: string;
  answer: string;
  citations: string[];
  ungrounded: string[];
  proposals: { action_id: string; action: string; lei: string; sanction_id: string | null; created: boolean; status: string }[];
  tool_calls: { name: string; is_error: boolean }[];
  error: string | null;
  insufficient: boolean;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const q = encodeURIComponent;

export const api = {
  summary: () => request<Summary>("/api/summary"),
  search: (text: string) => request<{ matches: SearchMatch[] }>(`/api/search?q=${q(text)}`),
  risks: (limit = 15) => request<{ companies: RiskRow[] }>(`/api/risks?limit=${limit}`),
  reviewQueue: () =>
    request<{ candidates: { company_ref: string; company_name: string; sanction_ref: string; sanction_name: string; score: number; country_relation: string }[] }>(
      "/api/review-queue",
    ),
  company: (lei: string) => request<Company>(`/api/companies/${q(lei)}`),
  relationships: (lei: string, relation: string) =>
    request<Relationships>(`/api/companies/${q(lei)}/relationships/${relation}`),
  news: (lei: string) => request<{ articles: Article[]; total: number }>(`/api/companies/${q(lei)}/news`),
  graph: (lei: string, depth: number) => request<GraphPayload>(`/api/companies/${q(lei)}/graph?depth=${depth}`),
  scenario: (lei: string, severity: number) =>
    request<Scenario>(`/api/companies/${q(lei)}/scenario?severity=${severity}`),
  actions: (status?: string) => request<{ actions: Action[] }>(`/api/actions${status ? `?status=${status}` : ""}`),
  decide: (id: string, approve: boolean, by: string, note: string | null) =>
    request<Action>(`/api/actions/${q(id)}/decision`, {
      method: "POST",
      body: JSON.stringify({ approve, by, note }),
    }),
  ask: (question: string, allowActions: boolean) =>
    request<AskResult>("/api/ask", {
      method: "POST",
      body: JSON.stringify({ question, allow_actions: allowActions }),
    }),
};

/** "lei:XYZ" -> "XYZ" */
export const leiOf = (ref: string) => ref.replace(/^lei:/, "");
