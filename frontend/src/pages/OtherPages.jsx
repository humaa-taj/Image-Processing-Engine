// The minimal System page (backend health check).
import { WorkspaceLayout } from "../components/Frame.jsx";
import { useHealth } from "../lib/useHealth.js";

export function SystemPage() {
  const { loading, data, error } = useHealth();
  return (
    <WorkspaceLayout>
      <div className="mb-8">
        <span className="label mb-3 inline-block rounded-full border-2 border-edge bg-butter px-3.5 py-0.5 !text-ink">System</span>
        <h1 className="font-serif text-[46px] leading-[1.08] font-semibold italic text-ink">Model status</h1>
        <p className="mt-2.5 text-[16px] text-muted">Live result of the backend health check (GET /api/health).</p>
      </div>
      {loading && <p className="label">Checking…</p>}
      {error && <p className="text-[14px] font-bold text-danger">{error}</p>}
      {data && (
        <div className="overflow-hidden rounded-panel border-2 border-edge bg-panel shadow-pop">
          <div className="flex gap-10 border-b-2 border-edge bg-mint px-7 py-5">
            <div><div className="label !text-ink-2">Status</div><div className="num mt-1 text-ink">{data.status}</div></div>
            <div><div className="label !text-ink-2">Models loaded</div><div className="num mt-1 text-ink">{data.models_loaded} / {data.models_expected}</div></div>
            {data.onnxruntime && <div><div className="label !text-ink-2">ONNX Runtime</div><div className="num mt-1 text-ink">{data.onnxruntime}</div></div>}
          </div>
          {Object.entries(data.models || {}).map(([name, m]) => (
            <div key={name} className="flex items-center gap-4 border-b-2 border-dashed border-edge/30 px-7 py-3.5 last:border-0">
              <span className={`h-3 w-3 shrink-0 rounded-full border-2 border-edge ${m.loaded ? "bg-ok" : "bg-faint"}`} />
              <span className="flex-1 text-[14px] font-bold text-ink-2">{m.description}</span>
              <span className="num text-[12px] text-muted">{m.file}</span>
              <span className="num w-20 text-right text-[12px] text-muted">{m.size_mb ? `${m.size_mb} MB` : ""}</span>
              <span className="num w-56 text-right text-[12px] text-muted">{m.loaded ? "loaded" : m.error}</span>
            </div>
          ))}
        </div>
      )}
    </WorkspaceLayout>
  );
}
