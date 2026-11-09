import { useEffect, useState } from "react";
import { leiOf } from "../api";

/** Load data for a component; re-runs when `deps` change. */
export function useLoad<T>(load: () => Promise<T>, deps: unknown[]): { data: T | null; error: string | null; loading: boolean } {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({
    data: null,
    error: null,
    loading: true,
  });
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    load()
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((e: Error) => live && setState({ data: null, error: e.message, loading: false }));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export function Loading({ error, loading }: { error: string | null; loading: boolean }) {
  if (error) return <p className="error" role="alert">{error}</p>;
  if (loading) return <p className="muted">Loading…</p>;
  return null;
}

/** A graph ref rendered as a link: companies open in the explorer, news opens its URL. */
export function RefLink({ refId, label, url }: { refId: string; label?: string; url?: string }) {
  const [kind] = refId.split(":", 1);
  const text = label ?? refId;
  if (kind === "lei") return <a href={`#/company/${leiOf(refId)}`}>{text}</a>;
  if (url) return <a href={url} target="_blank" rel="noreferrer">{text}</a>;
  return <span className="mono">{text}</span>;
}

export function Flags({ sanctioned, review, watchlist }: { sanctioned?: boolean; review?: boolean; watchlist?: boolean }) {
  return (
    <>
      {sanctioned && <span className="status critical">Sanctioned</span>}{" "}
      {review && <span className="status warning">Review candidate</span>}{" "}
      {watchlist && <span className="status neutral">Watchlist</span>}
    </>
  );
}
