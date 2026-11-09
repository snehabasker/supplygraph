import { useEffect, useState } from "react";
import { api } from "./api";
import ActionsPage from "./components/ActionsPage";
import AskPage from "./components/AskPage";
import Dashboard from "./components/Dashboard";
import Explorer from "./components/Explorer";

// Hash routes: #/ (dashboard), #/company/<LEI>, #/actions, #/ask
function useRoute(): string[] {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => setRoute(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export const go = (path: string) => {
  window.location.hash = `#/${path}`;
};

export default function App() {
  const route = useRoute();
  const page = route[0] ?? "";
  const [proposed, setProposed] = useState<number | null>(null);

  // The open-proposal count in the nav refreshes on every navigation.
  useEffect(() => {
    api.summary().then((s) => setProposed(s.proposed_actions)).catch(() => setProposed(null));
  }, [route.join("/")]);

  const nav = [
    { path: "", label: "Dashboard" },
    { path: "company", label: "Explorer" },
    { path: "actions", label: "Actions", badge: proposed },
    { path: "ask", label: "Ask" },
  ];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          SupplyGraph<small>supplier risk explorer</small>
        </div>
        <nav className="nav" aria-label="Main">
          {nav.map((n) => (
            <a key={n.path} href={`#/${n.path}`} aria-current={page === n.path ? "page" : undefined}>
              {n.label}
              {n.badge ? <span className="badge" aria-label={`${n.badge} awaiting decision`}>{n.badge}</span> : null}
            </a>
          ))}
        </nav>
      </header>
      <main>
        {page === "" && <Dashboard />}
        {page === "company" && <Explorer lei={route[1]} view={route[2]} />}
        {page === "actions" && <ActionsPage onChange={() => api.summary().then((s) => setProposed(s.proposed_actions))} />}
        {page === "ask" && <AskPage />}
        {!["", "company", "actions", "ask"].includes(page) && <p>Page not found. <a href="#/">Dashboard</a></p>}
      </main>
    </div>
  );
}
