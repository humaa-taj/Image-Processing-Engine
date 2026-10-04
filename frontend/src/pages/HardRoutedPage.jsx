// Workspace 02: Hard-Routed Restoration (Task 2). Classifier -> exactly one specialist (or identity).
import { useState } from "react";
import { Button, CorruptionPanel, ErrorBanner, ErrorLegend, ImagePanel, ProgressStatus, ResultStrip, RoutingBars, UploadPanel } from "../components/Controls.jsx";
import { PageHeader, WorkspaceLayout } from "../components/Frame.jsx";
import { FrameIcon, WandIcon } from "../components/Icons.jsx";
import { api } from "../lib/api.js";
import { useRestoration } from "../lib/useRestoration.js";
import { settingsItems } from "./UniversalPage.jsx";

const CLASS_NAMES = { clean: "Clean", salt: "Salt-and-pepper", blur: "Gaussian blur", occlusion: "Rectangular occlusion" };
const EXPERT_NAMES = { identity: "Identity (bypass)", "salt expert": "Salt-and-pepper expert", "blur expert": "Blur expert", "occlusion expert": "Occlusion expert" };

export default function HardRoutedPage() {
  const s = useRestoration(api.restoreHard);
  const r = s.result;
  const [view, setView] = useState("restored");            // "restored" | "error"
  const showError = view === "error" && r?.reference_available;

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
      <PageHeader number={2} title="Hard-Routed Restoration" description="A classifier picks one specialist autoencoder for each image." />
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
          empty={s.phase === "restoring" ? "Running classifier and expert" : "Awaiting restoration"} icon={<WandIcon width={22} height={22} />}>
          {viewSwitch}
        </ImagePanel>

        <section className="flex min-w-0 flex-col">
          <div className="label mb-2.5 flex h-6 items-center !text-[14px] !font-extrabold !text-ink">Routing</div>
          <div className="flex h-[330px] flex-col rounded-panel border-2 border-edge bg-panel p-6 shadow-pop">
            {r ? (
              <>
                <RoutingBars values={Object.fromEntries(Object.entries(r.probabilities).map(([k, v]) => [CLASS_NAMES[k], v]))}
                  highlight={CLASS_NAMES[r.predicted_class]} />
                <div className="mt-auto flex flex-col gap-2 border-t-2 border-dashed border-edge/40 pt-4">
                  <div className="flex justify-between"><span className="label">Predicted</span><span className="text-[14px] font-extrabold text-ink">{CLASS_NAMES[r.predicted_class]}</span></div>
                  <div className="flex justify-between"><span className="label">Selected expert</span><span className="text-[14px] font-extrabold text-accent">{EXPERT_NAMES[r.selected_expert]}</span></div>
                </div>
              </>
            ) : (
              <div className="flex flex-1 items-center justify-center">
                <span className={`label text-center ${s.phase === "restoring" ? "animate-pulse" : ""}`}>
                  {s.phase === "restoring" ? "Classifying" : "Classifier probabilities appear here"}
                </span>
              </div>
            )}
          </div>
        </section>
      </div>

      {s.busy && <ProgressStatus startedAt={s.startedAt} label={s.phase === "applying" ? "Applying corruption" : "Running classifier and expert"} />}
      {r && !s.busy && (
        <ResultStrip downloadUrl={r.output_image} downloadName="hard_routed_restoration.png"
          items={[...settingsItems(r, s.alreadyCorrupted),
            ["Route", EXPERT_NAMES[r.selected_expert]],
            ["Inference time", `${r.timing_ms.inference.toFixed(1)} ms`]]} />
      )}
      {r && !r.reference_available && !s.busy && (
        <p className="mt-4 text-[13px] font-bold text-muted">No error map: the clean original of an already-corrupted upload is unknown.</p>
      )}
    </WorkspaceLayout>
  );
}
