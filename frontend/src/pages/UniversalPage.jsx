// Workspace 01: Universal Restoration (Task 1). One autoencoder for every corruption.
import { Button, CorruptionPanel, ErrorBanner, ErrorLegend, ImagePanel, ProgressStatus, ResultStrip, UploadPanel } from "../components/Controls.jsx";
import { PageHeader, WorkspaceLayout } from "../components/Frame.jsx";
import { FrameIcon, WandIcon } from "../components/Icons.jsx";
import { api } from "../lib/api.js";
import { corruptionLabel, LEVELS, paramsText } from "../lib/corruptions.js";
import { useRestoration } from "../lib/useRestoration.js";

/** The settings row shown under a result (only real values from the response). */
export function settingsItems(r, alreadyCorrupted) {
  const c = r.corruption;
  return [
    ["Corruption", c ? corruptionLabel(c.type) : alreadyCorrupted ? "None (already corrupted)" : "None"],
    ["Severity", c ? LEVELS.find((l) => l.id === c.level).label : "—"],
    ["Parameters", c ? paramsText(c.type, c.level) : "—"],
  ];
}

export default function UniversalPage() {
  const s = useRestoration(api.restoreUniversal);
  const r = s.result;
  const showError = r?.reference_available;   // only when the clean original is really known

  return (
    <WorkspaceLayout>
      <PageHeader number={1} title="Universal Restoration" description="One autoencoder that repairs noise, blur and missing regions." />
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

      <div className={`mt-9 grid gap-7 ${showError ? "grid-cols-3" : "grid-cols-2"}`}>
        <ImagePanel label="Input" src={s.inputImage} empty="Awaiting source image" icon={<FrameIcon width={22} height={22} />} />
        <ImagePanel label="Restored" src={r?.output_image} busy={s.phase === "restoring"}
          empty={s.phase === "restoring" ? "Running universal autoencoder" : "Awaiting restoration"} icon={<WandIcon width={22} height={22} />} />
        {showError && (
          <ImagePanel label="Error map" src={r.error_map_image} footer={<ErrorLegend />}>
            <span className="num normal-case tracking-normal">MAE {r.mean_abs_error.toFixed(4)}</span>
          </ImagePanel>
        )}
      </div>

      {s.busy && <ProgressStatus startedAt={s.startedAt} label={s.phase === "applying" ? "Applying corruption" : "Running universal autoencoder"} />}
      {r && !s.busy && (
        <ResultStrip downloadUrl={r.output_image} downloadName="universal_restoration.png"
          items={[...settingsItems(r, s.alreadyCorrupted), ["Inference time", `${r.timing_ms.inference.toFixed(1)} ms`]]} />
      )}
      {r && !showError && !s.busy && (
        <p className="mt-4 text-[13px] font-bold text-muted">No error map: the clean original of an already-corrupted upload is unknown.</p>
      )}
    </WorkspaceLayout>
  );
}
