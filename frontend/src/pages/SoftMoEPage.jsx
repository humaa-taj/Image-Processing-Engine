// Workspace 03: Soft Mixture-of-Experts Restoration (Task 3). A gate blends all four branches.
// Every number shown (weights, top contributors, time) comes from the backend response.
import { useState } from "react";
import { Button, CorruptionPanel, ErrorBanner, ErrorLegend, ImagePanel, ProgressStatus, ResultStrip, RoutingBars, RoutingDiagram, UploadPanel } from "../components/Controls.jsx";
import { PageHeader, WorkspaceLayout } from "../components/Frame.jsx";
import { FrameIcon, WandIcon } from "../components/Icons.jsx";
import { api } from "../lib/api.js";
import { useRestoration } from "../lib/useRestoration.js";
import { settingsItems } from "./UniversalPage.jsx";

// backend branch key -> display name (fixed gate order: identity, salt, blur, occlusion)
const BRANCH_NAMES = { identity: "Identity", "salt expert": "Salt-and-pepper", "blur expert": "Blur", "occlusion expert": "Occlusion" };

export default function SoftMoEPage() {
  const s = useRestoration(api.restoreSoft);
  const r = s.result;
  const [view, setView] = useState("restored");
  const showError = view === "error" && r?.reference_available;
  const named = r ? Object.entries(r.weights).map(([k, v]) => [BRANCH_NAMES[k], v]) : [];

  const viewSwitch = r?.reference_available && (
    <span className="flex gap-1 rounded-full border-2 border-edge bg-white p-0.5 normal-case tracking-normal">
      {[["restored", "Restored"], ["error", "Error map"]].map(([id, text]) => (
        <button key={id} onClick={() => setView(id)}
          className={`rounded-full px-3 py-0.5 text-[12px] font-extrabold ${view === id ? "bg-accent text-on-accent" : "text-ink-2 hover:bg-soft"}`}>{text}</button>
      ))}
    </span>
  );

  return (
    <WorkspaceLayout>
      <PageHeader number={3} title="Soft Mixture-of-Experts Restoration" description="A gate blends all specialists with continuous weights." />
      <ErrorBanner message={s.error} onClose={() => s.setError(null)} />

      <div className="grid grid-cols-2 gap-7">
        <UploadPanel source={s.source} onSource={s.setSource} disabled={s.busy} onError={s.setError}
          toggle={{ label: "Image is already corrupted", checked: s.alreadyCorrupted, onChange: s.setAlreadyCorrupted }} />
        <CorruptionPanel corruption={s.corruption} level={s.level} onCorruption={s.setCorruption} onLevel={s.setLevel}
          locked={s.alreadyCorrupted} disabled={s.busy}>
          <Button onClick={s.applyCorruption} disabled={!s.canApply}>{s.phase === "applying" ? "Applying…" : "Apply corruption"}</Button>
          <Button variant="primary" onClick={s.restore} disabled={!s.canRestore}>{s.phase === "restoring" ? "Restoring…" : "Restore"}</Button>
        </CorruptionPanel>
      </div>

      <div className="mt-9 grid grid-cols-3 gap-7">
        <ImagePanel label="Input" src={s.inputImage} empty="Awaiting source image" icon={<FrameIcon width={22} height={22} />} />
        <ImagePanel label={showError ? "Error map" : "Restored"} src={showError ? r.error_map_image : r?.output_image}
          busy={s.phase === "restoring"} footer={showError ? <ErrorLegend /> : null}
          empty={s.phase === "restoring" ? "Running gate and experts" : "Awaiting restoration"} icon={<WandIcon width={22} height={22} />}>
          {viewSwitch}
        </ImagePanel>
        <section className="flex min-w-0 flex-col">
          <div className="label mb-2.5 flex h-6 items-center !text-[14px] !font-extrabold !text-ink">Expert weights</div>
          <div className="flex h-[330px] flex-col rounded-panel border-2 border-edge bg-panel p-6 shadow-pop">
            {r ? (
              <>
                <RoutingBars values={Object.fromEntries(named)} highlight={BRANCH_NAMES[r.dominant_branch]} />
                <div className="mt-auto border-t-2 border-dashed border-edge/40 pt-4">
                  <div className="label">Top contributors</div>
                  <div className="mt-1.5 text-[14px] font-extrabold text-ink">{r.top_contributors.map((b) => BRANCH_NAMES[b]).join(", ")}</div>
                </div>
              </>
            ) : (
              <div className="flex flex-1 items-center justify-center">
                <span className={`label text-center ${s.phase === "restoring" ? "animate-pulse" : ""}`}>
                  {s.phase === "restoring" ? "Computing weights" : "Routing weights appear here"}
                </span>
              </div>
            )}
          </div>
        </section>
      </div>

      {s.busy && <ProgressStatus startedAt={s.startedAt} label={s.phase === "applying" ? "Applying corruption" : "Running gate and experts"} />}
      {r && !s.busy && (
        <>
          <section className="mt-7">
            <div className="label mb-2.5 !text-[14px] !font-extrabold !text-ink">Routing</div>
            <div className="rounded-panel border-2 border-edge bg-panel px-7 py-5 shadow-pop">
              <RoutingDiagram weights={named} dominant={BRANCH_NAMES[r.dominant_branch]} />
            </div>
          </section>
          <ResultStrip downloadUrl={r.output_image} downloadName="soft_moe_restoration.png"
            items={[...settingsItems(r, s.alreadyCorrupted), ["Inference time", `${r.timing_ms.inference.toFixed(1)} ms`]]} />
          {!r.reference_available && (
            <p className="mt-4 text-[13px] font-bold text-muted">No error map: the clean original of an already-corrupted upload is unknown.</p>
          )}
        </>
      )}
    </WorkspaceLayout>
  );
}
