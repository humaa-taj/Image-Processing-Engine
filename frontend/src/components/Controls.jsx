// Workspace building blocks shared by all restoration workspaces.
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api.js";
import { CORRUPTIONS, LEVELS, MAX_UPLOAD_MB, paramsText, validateFile } from "../lib/corruptions.js";
import { ChevronIcon, CloseIcon, DownloadIcon, SamplesIcon, UploadIcon } from "./Icons.jsx";

export function Panel({ title, className = "", children }) {
  return (
    <section className={`flex flex-col ${className}`}>
      {title && (
        <h2 className="mb-3 flex items-center gap-2.5 text-[17px] font-extrabold text-ink">
          <span className="h-3 w-3 rounded-full border-2 border-edge bg-accent" aria-hidden />
          {title}
        </h2>
      )}
      <div className="flex-1 rounded-panel border-2 border-edge bg-panel p-6 shadow-pop">{children}</div>
    </section>
  );
}

export function Button({ variant = "secondary", className = "", ...props }) {
  const look = variant === "primary"
    ? "border-edge bg-accent text-on-accent hover:bg-accent-strong disabled:border-edge/30 disabled:bg-[#f4b3cf] disabled:text-white"
    : "border-edge bg-white text-ink hover:bg-soft disabled:border-edge/30 disabled:bg-inset disabled:text-faint";
  return (
    <button
      className={`inline-flex h-12 items-center justify-center gap-2 rounded-full border-2 px-6 text-[14px] font-extrabold shadow-pop-sm transition-all active:translate-x-[3px] active:translate-y-[3px] active:shadow-none disabled:cursor-not-allowed disabled:shadow-none ${look} ${className}`}
      {...props}
    />
  );
}

export function Toggle({ checked, onChange, label, disabled }) {
  return (
    <label className={`flex items-center justify-between rounded-control border-2 border-edge bg-inset px-4 py-3 text-[14px] font-bold ${disabled ? "text-faint" : "text-ink"}`}>
      {label}
      <button
        type="button" role="switch" aria-checked={checked} disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative h-7 w-12 shrink-0 rounded-full border-2 border-edge transition-colors ${checked ? "bg-accent" : "bg-white"} disabled:opacity-50`}
      >
        <span className={`absolute top-[3px] h-4 w-4 rounded-full border-2 border-edge bg-white transition-all ${checked ? "left-[22px]" : "left-[3px] !bg-edge"}`} />
      </button>
    </label>
  );
}

/** Sample chooser: a floating layer listing the bundled clean images from /api/samples. */
function SamplePicker({ kind, onPick, onClose }) {
  const [state, setState] = useState({ loading: true, items: [], error: null });
  useEffect(() => {
    api.samples().then((s) => setState({ loading: false, items: s[kind] || [], error: null }))
      .catch((e) => setState({ loading: false, items: [], error: e.message }));
  }, [kind]);
  return (
    <div className="glass absolute inset-x-0 top-11 z-20 rounded-panel border-2 border-edge p-5 shadow-pop">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-[14px] font-extrabold text-ink">Choose a sample</span>
        <button onClick={onClose} className="rounded-full border-2 border-edge p-1 text-ink hover:bg-soft" aria-label="Close"><CloseIcon width={14} height={14} /></button>
      </div>
      {state.loading && <p className="label">Loading…</p>}
      {state.error && <p className="text-[13px] font-bold text-danger">{state.error}</p>}
      <div className="grid max-h-64 grid-cols-6 gap-2.5 overflow-y-auto p-1">
        {state.items.map((s) => (
          <button key={s.name} onClick={() => onPick(s)} title={s.name.split("/").pop()}
            className="aspect-square overflow-hidden rounded-[14px] border-2 border-edge hover:-translate-y-0.5 hover:border-accent hover:shadow-pop-sm">
            <img src={s.url} alt={s.name} className="h-full w-full object-cover" />
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * Source panel: drop zone / browse, "Choose a sample", Clear, and an optional toggle.
 * source = { file?, sample?, name, preview } or null.
 * toggle = { label, checked, onChange } or null (e.g. "Image is already corrupted").
 * tabs   = optional element shown above the drop zone (Face-to-Sketch: Upload | Webcam).
 * children = optional element that REPLACES the drop zone (Face-to-Sketch: the webcam view).
 */
export function UploadPanel({ source, onSource, toggle, tabs, children, disabled, sampleKind = "pets", onError }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const [picking, setPicking] = useState(false);

  const takeFile = (file) => {
    if (!file) return;
    const err = validateFile(file);
    if (err) return onError(err);
    onSource({ file, name: file.name, preview: URL.createObjectURL(file) });
  };

  return (
    <Panel title="Source" className="relative">
      <div className="flex h-full flex-col gap-4">
        {tabs}
        {children ?? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); if (!disabled) takeFile(e.dataTransfer.files[0]); }}
          className={`relative flex min-h-[176px] flex-1 flex-col items-center justify-center rounded-control border-2 border-dashed transition-colors ${
            drag ? "border-accent bg-soft" : "border-edge/60 bg-inset"} ${disabled ? "opacity-60" : ""}`}
        >
          {source && (
            <button onClick={() => onSource(null)} disabled={disabled}
              className="label absolute right-3 top-3 rounded-full border-2 border-edge bg-white px-3 py-0.5 !text-ink hover:bg-soft">Clear</button>
          )}
          {source ? (
            <div className="flex flex-col items-center gap-2.5">
              <img src={source.preview} alt="" className="h-20 w-20 rounded-[18px] border-2 border-edge object-cover shadow-pop-sm" />
              <span className="num max-w-[240px] truncate text-[12px] text-ink-2">{source.name}</span>
            </div>
          ) : (
            <>
              <button onClick={() => inputRef.current?.click()} disabled={disabled}
                className="mb-3 flex h-12 w-12 items-center justify-center rounded-full border-2 border-edge bg-lilac text-ink shadow-pop-sm hover:bg-white" aria-label="Browse">
                <UploadIcon />
              </button>
              <button onClick={() => inputRef.current?.click()} disabled={disabled} className="text-[15px] font-extrabold text-ink hover:underline">
                Drop an image or browse
              </button>
              <span className="num mt-1.5 text-[12px] text-muted">JPG or PNG, up to {MAX_UPLOAD_MB} MB</span>
              <button onClick={() => setPicking(true)} disabled={disabled}
                className="mt-4 inline-flex items-center gap-2 rounded-full border-2 border-edge bg-white px-4 py-1.5 text-[13px] font-bold text-ink hover:bg-soft">
                <SamplesIcon width={14} height={14} /> Choose a sample
              </button>
            </>
          )}
          <input ref={inputRef} type="file" accept="image/jpeg,image/png" className="hidden"
            onChange={(e) => { takeFile(e.target.files[0]); e.target.value = ""; }} />
        </div>
        )}
        {toggle && <Toggle label={toggle.label} checked={toggle.checked} onChange={toggle.onChange} disabled={disabled} />}
      </div>
      {picking && (
        <SamplePicker kind={sampleKind} onClose={() => setPicking(false)}
          onPick={(s) => { setPicking(false); onSource({ sample: s.name, name: s.name.split("/").pop(), preview: s.url }); }} />
      )}
    </Panel>
  );
}

/** Corruption panel: type, intensity, exact parameters, and the action buttons passed as children. */
export function CorruptionPanel({ corruption, level, onCorruption, onLevel, locked, disabled, children }) {
  const off = locked || disabled;
  return (
    <Panel title="Corruption">
      <div className="flex h-full flex-col gap-4">
        <div className={`flex flex-col gap-4 ${off ? "opacity-60" : ""}`}>
        <div>
          <div className="label mb-2">Type</div>
          <div className="relative">
            <select value={corruption} disabled={off} onChange={(e) => onCorruption(e.target.value)}
              className="h-12 w-full appearance-none rounded-full border-2 border-edge bg-white px-5 text-[14px] font-bold text-ink disabled:cursor-not-allowed">
              {CORRUPTIONS.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
            </select>
            <ChevronIcon className="pointer-events-none absolute right-4 top-4 text-ink" />
          </div>
        </div>
        <div>
          <div className="label mb-2">Intensity</div>
          <div className="grid grid-cols-3 gap-1.5 rounded-full border-2 border-edge bg-inset p-1.5">
            {LEVELS.map((l) => (
              <button key={l.id} disabled={off || corruption === "none"} onClick={() => onLevel(l.id)}
                className={`h-9 rounded-full text-[13px] font-extrabold transition-colors disabled:cursor-not-allowed ${
                  level === l.id ? "bg-accent text-on-accent" : "text-ink-2 hover:bg-white"}`}>
                {l.label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center justify-between rounded-control border-2 border-edge bg-inset px-4 py-2.5">
          <span className="label">Parameters</span>
          <span className="num text-[12.5px] text-ink">{locked ? "—" : paramsText(corruption, level)}</span>
        </div>
        </div>
        <div className="mt-auto grid grid-cols-2 gap-3">{children}</div>
      </div>
    </Panel>
  );
}

/** One labelled image area with corner dots; shows a placeholder message when empty. */
export function ImagePanel({ label, src, empty, icon, busy, children, footer }) {
  return (
    <section className="flex min-w-0 flex-col">
      <div className="label mb-2.5 flex h-6 items-center justify-between !text-[14px] !font-extrabold !text-ink">{label}{children}</div>
      <div className="relative h-[330px] overflow-hidden rounded-panel border-2 border-edge bg-inset shadow-pop">
        {["left-3.5 top-3.5", "right-3.5 top-3.5", "left-3.5 bottom-3.5", "right-3.5 bottom-3.5"].map((c) => (
          <span key={c} className={`absolute z-10 h-2.5 w-2.5 rounded-full border-2 border-edge bg-butter ${c}`} aria-hidden />
        ))}
        {src ? (
          <img src={src} alt={label} className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-muted">
            {icon}
            <span className={`label ${busy ? "animate-pulse" : ""}`}>{empty}</span>
          </div>
        )}
      </div>
      {footer}
    </section>
  );
}

/** Low -> high legend for error maps: seven flat steps sampled from the backend's error_map colour ramp. */
const LEGEND_STEPS = ["#0e0f11", "#323c49", "#566982", "#7a96ba", "#a0b4ce", "#c6d2e1", "#ecf0f5"];
export function ErrorLegend() {
  return (
    <div className="mt-3 flex items-center gap-2.5">
      <span className="label">Low</span>
      <span className="flex h-3 flex-1 overflow-hidden rounded-full border-2 border-edge">
        {LEGEND_STEPS.map((c) => <span key={c} className="h-full flex-1" style={{ background: c }} />)}
      </span>
      <span className="label">High</span>
    </div>
  );
}

/** Indeterminate progress + the REAL elapsed time (we never show a fake percentage). */
export function ProgressStatus({ label, startedAt }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 100); return () => clearInterval(id); }, []);
  return (
    <div className="mt-7">
      <div className="h-3 overflow-hidden rounded-full border-2 border-edge bg-white">
        <div className="progress-indeterminate h-full w-2/5 bg-accent" />
      </div>
      <div className="mt-2.5 flex justify-between">
        <span className="label">{label}</span>
        <span className="label num">{((now - startedAt) / 1000).toFixed(1)} s</span>
      </div>
    </div>
  );
}

/** Bottom strip with the run's real settings + Download result. items = [[label, value], ...]. */
export function ResultStrip({ items, downloadUrl, downloadName = "restored.png", downloadLabel = "Download result" }) {
  return (
    <div className="mt-7 flex flex-wrap items-center gap-x-10 gap-y-3 rounded-panel border-2 border-edge bg-butter px-7 py-5 shadow-pop">
      {items.map(([k, v]) => (
        <div key={k}>
          <div className="label !text-ink-2">{k}</div>
          <div className="num mt-0.5 text-[15px] text-ink">{v}</div>
        </div>
      ))}
      {downloadUrl && (
        <a href={downloadUrl} download={downloadName} className="ml-auto">
          <Button variant="secondary" tabIndex={-1}><DownloadIcon /> {downloadLabel}</Button>
        </a>
      )}
    </div>
  );
}

export function ErrorBanner({ message, onClose }) {
  if (!message) return null;
  return (
    <div role="alert" className="mb-6 flex items-center justify-between rounded-control border-2 border-danger bg-white px-5 py-3 text-[14px] font-bold text-danger shadow-pop-sm">
      {message}
      <button onClick={onClose} className="text-danger hover:text-ink" aria-label="Dismiss"><CloseIcon /></button>
    </div>
  );
}

/** Horizontal bars for the 4 classifier probabilities (or mixture weights). values = {name: number}. */
export function RoutingBars({ values, highlight }) {
  const rows = Object.entries(values).sort((a, b) => b[1] - a[1]);
  return (
    <div className="flex flex-col gap-4">
      {rows.map(([name, v]) => (
        <div key={name}>
          <div className="mb-1.5 flex justify-between">
            <span className={`text-[13px] ${name === highlight ? "font-extrabold text-ink" : "font-bold text-muted"}`}>{name}</span>
            <span className="num text-[13px] text-ink-2">{(v * 100).toFixed(1)}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full border-2 border-edge bg-white">
            <div className={`h-full rounded-full ${name === highlight ? "bg-accent" : "bg-lilac"}`} style={{ width: `${Math.max(v * 100, 0.5)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Gate -> 4 branches diagram. Each curve's thickness and brightness are proportional to that branch's
 * weight from the backend response; the dominant branch is highlighted. weights = [[name, w], ...] in a fixed order.
 */
export function RoutingDiagram({ weights, dominant }) {
  const W = 1000, H = 230, gateX = 230, nodeX = 560, nodeW = 200;
  const ys = weights.map((_, i) => 30 + i * ((H - 60) / (weights.length - 1)));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-[230px] w-full" role="img" aria-label="Gate routing weights">
      {weights.map(([name, w], i) => (
        <path key={name} d={`M ${gateX + 90} ${H / 2} C ${gateX + 230} ${H / 2}, ${nodeX - 140} ${ys[i]}, ${nodeX} ${ys[i]}`}
          fill="none" stroke="var(--color-accent)" strokeWidth={1 + 9 * w} strokeOpacity={0.22 + 0.78 * w} strokeLinecap="round" />
      ))}
      <rect x={gateX - 90} y={H / 2 - 22} width={180} height={44} rx={22} fill="var(--color-butter)" stroke="var(--color-edge)" strokeWidth="2" />
      <text x={gateX} y={H / 2 + 5} textAnchor="middle" fill="var(--color-ink)" fontSize="15" fontWeight="800" fontFamily="var(--font-sans)">Gate</text>
      {weights.map(([name, w], i) => {
        const top = name === dominant;
        return (
          <g key={name}>
            <rect x={nodeX} y={ys[i] - 17} width={nodeW} height={34} rx={17}
              fill={top ? "var(--color-soft)" : "var(--color-panel)"} stroke={top ? "var(--color-accent)" : "var(--color-edge)"} strokeWidth="2" />
            <text x={nodeX + 18} y={ys[i] + 5} fill={top ? "var(--color-ink)" : "var(--color-muted)"} fontSize="14" fontWeight={top ? "800" : "700"} fontFamily="var(--font-sans)">{name}</text>
            <text x={nodeX + nodeW + 16} y={ys[i] + 5} fill="var(--color-ink-2)" fontSize="13" fontWeight="700" fontFamily="var(--font-mono)">{(w * 100).toFixed(1)}%</text>
          </g>
        );
      })}
    </svg>
  );
}
