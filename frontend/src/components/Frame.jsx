// The app frame: TopBar, Sidebar, PageHeader and the WorkspaceLayout that combines them.
// The design rules (wordmark, status badge, the four tab names, page header) are implemented ONLY here,
// so every page is identical by construction.
import { Link, NavLink } from "react-router-dom";
import { MLFLOW_URL, MOCK } from "../lib/api.js";
import { useHealth } from "../lib/useHealth.js";

export const WORKSPACES = [
  { path: "/universal", name: "Universal Restoration" },
  { path: "/hard-routed", name: "Hard-Routed Restoration" },
  { path: "/soft-moe", name: "Soft Mixture-of-Experts Restoration" },
  { path: "/face-to-sketch", name: "Face-to-Sketch Generator" },
];

export function Wordmark() {
  return (
    <Link to="/" className="flex shrink-0 items-center gap-3 font-serif text-[22px] font-semibold italic text-ink whitespace-nowrap hover:text-accent">
      <span className="h-4 w-4 rounded-full border-2 border-edge bg-dot" aria-hidden />
      Image Processing Engine
    </Link>
  );
}

/** Top-right status. Driven by /api/health; in mock mode it is replaced by the MOCK DATA badge. */
export function StatusBadge() {
  const { loading, data, error } = useHealth();
  if (MOCK) {
    return <span className="label rounded-full border-2 border-edge bg-butter px-3.5 py-1 !text-ink">Mock data</span>;
  }
  let dot = "bg-faint";
  let text = "Checking models";
  if (error) text = "Backend offline";
  else if (!loading && data) {
    const all = data.models_loaded === data.models_expected;
    dot = all ? "bg-ok" : "bg-faint";
    text = all ? "Models loaded" : `${data.models_loaded}/${data.models_expected} models loaded`;
  }
  return (
    <Link to="/system" className="label flex items-center gap-2 whitespace-nowrap rounded-full border-2 border-edge bg-mint px-3.5 py-1 !text-ink hover:bg-white" title="Open System status">
      <span className={`h-2.5 w-2.5 rounded-full border border-edge ${dot}`} aria-hidden />
      {text}
    </Link>
  );
}

export function TopBar({ links = false }) {
  return (
    <header className="flex h-[72px] shrink-0 items-center justify-between border-b-2 border-edge bg-bar px-[70px] max-lg:px-6">
      <Wordmark />
      <div className="flex items-center gap-6">
        {links && (
          <nav className="flex items-center gap-2 text-[15px] font-bold text-ink max-md:hidden">
            <Link to="/universal" className="rounded-full px-4 py-1.5 hover:bg-soft">Workspaces</Link>
            <Link to="/system" className="rounded-full px-4 py-1.5 hover:bg-soft">System</Link>
            <a href={MLFLOW_URL} target="_blank" rel="noreferrer" className="rounded-full px-4 py-1.5 hover:bg-soft">Experiments</a>
          </nav>
        )}
        <StatusBadge />
      </div>
    </header>
  );
}

export function Sidebar() {
  const tab = ({ isActive }) =>
    `relative block whitespace-nowrap rounded-full border-2 px-5 py-2.5 text-[14px] font-bold transition-all ${
      isActive
        ? "border-edge bg-accent text-on-accent shadow-pop-sm"
        : "border-transparent text-ink-2 hover:border-edge hover:bg-white hover:text-ink"
    }`;
  const quiet = "block rounded-full px-4 py-1.5 text-[13px] font-bold text-muted hover:bg-white hover:text-ink";
  return (
    <aside className="flex w-[296px] shrink-0 flex-col border-r-2 border-edge bg-side px-4 py-7">
      <nav className="flex flex-col gap-2.5">
        {WORKSPACES.map((w) => (
          <NavLink key={w.path} to={w.path} className={tab}>
            {w.name}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto flex flex-col gap-1 border-t-2 border-dashed border-edge/40 pt-4">
        <Link to="/" className={quiet}>Home</Link>
        <Link to="/system" className={quiet}>System</Link>
        <a href={MLFLOW_URL} target="_blank" rel="noreferrer" className={quiet}>Experiments</a>
      </div>
    </aside>
  );
}

export function PageHeader({ number, title, description }) {
  return (
    <div className="mb-8">
      <span className="label mb-3 inline-block rounded-full border-2 border-edge bg-butter px-3.5 py-0.5 !text-ink">Workspace {String(number).padStart(2, "0")}</span>
      <h1 className="font-serif text-[46px] leading-[1.08] font-semibold italic text-ink">{title}</h1>
      <p className="mt-2.5 text-[16px] text-muted">{description}</p>
    </div>
  );
}

/** Shared page width and margins for every workspace page. */
export function WorkspaceLayout({ children }) {
  return (
    <div className="flex h-full flex-col">
      <TopBar />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <main className="bg-grid min-w-0 flex-1 overflow-y-auto">
          <div className="max-w-[1240px] px-12 py-10">{children}</div>
        </main>
      </div>
    </div>
  );
}
