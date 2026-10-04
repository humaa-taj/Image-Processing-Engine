// Workspace 04: Face-to-Sketch Generator (Task 4). Photo (upload or webcam) + style -> sketch.
import { useEffect, useRef, useState } from "react";
import { Button, ErrorBanner, ImagePanel, Panel, ProgressStatus, ResultStrip, UploadPanel } from "../components/Controls.jsx";
import { PageHeader, WorkspaceLayout } from "../components/Frame.jsx";
import { ImageIcon, WandIcon } from "../components/Icons.jsx";
import { api } from "../lib/api.js";

/** Human-readable message for each way getUserMedia can fail. */
function cameraError(e) {
  if (!window.isSecureContext) return "The camera only works on a secure page (http://localhost or HTTPS).";
  switch (e?.name) {
    case "NotAllowedError":
    case "SecurityError": return "Camera access was denied. Allow the camera in the browser's address bar, then try again.";
    case "NotFoundError":
    case "OverconstrainedError": return "No camera was found on this device.";
    case "NotReadableError": return "The camera is in use by another application.";
    default: return `The camera could not be started${e?.message ? `: ${e.message}` : "."}`;
  }
}

/** Live webcam preview + Capture. Calls onCapture with a PNG File. The camera stops when this unmounts. */
function Webcam({ onCapture, disabled }) {
  const video = useRef(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    let s = null;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(cameraError(null));
      return undefined;
    }
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then((st) => { s = st; if (!active) { st.getTracks().forEach((t) => t.stop()); return; } setStream(st); if (video.current) video.current.srcObject = st; })
      .catch((e) => active && setError(cameraError(e)));
    return () => { active = false; s?.getTracks().forEach((t) => t.stop()); };   // always release the camera
  }, []);

  const capture = () => {
    const v = video.current;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((blob) => onCapture(new File([blob], "webcam.png", { type: "image/png" })), "image/png");
  };

  return (
    <div className="relative flex min-h-[176px] flex-1 flex-col items-center justify-center overflow-hidden rounded-control border-2 border-edge bg-inset">
      {error ? (
        <p className="max-w-[340px] px-4 text-center text-[13px] font-bold text-danger">{error}</p>
      ) : (
        <>
          {/* mirrored preview feels natural; the captured photo is not mirrored */}
          <video ref={video} autoPlay playsInline muted className="h-full max-h-[176px] w-full -scale-x-100 object-cover" />
          <Button variant="primary" onClick={capture} disabled={!stream || disabled} className="absolute bottom-3 h-10 text-[13px]">
            {stream ? "Capture photo" : "Starting camera…"}
          </Button>
        </>
      )}
    </div>
  );
}

export default function FaceToSketchPage() {
  const [mode, setMode] = useState("upload");            // "upload" | "webcam"
  const [source, setSourceState] = useState(null);
  const [style, setStyle] = useState(1);                  // default Style 1
  const [cropped, setCropped] = useState(false);          // true -> fit "stretch" (photo is already a face crop)
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [startedAt, setStartedAt] = useState(0);
  const [error, setError] = useState(null);

  const setSource = (s) => { setSourceState(s); setResult(null); setError(null); };
  const generate = async () => {
    setBusy(true); setError(null); setStartedAt(Date.now());
    try {
      setResult(await api.sketch(source, { style, fit: cropped ? "stretch" : "crop" }));
    } catch (e) {
      setError(`Sketch generation failed: ${e.message}`);
    }
    setBusy(false);
  };

  const tabs = (
    <div className="grid w-fit grid-cols-2 gap-1 rounded-full border-2 border-edge bg-inset p-1">
      {[["upload", "Upload"], ["webcam", "Webcam"]].map(([id, text]) => (
        <button key={id} onClick={() => setMode(id)} disabled={busy}
          className={`h-8 rounded-full px-5 text-[13px] font-extrabold ${mode === id ? "bg-accent text-on-accent" : "text-ink-2 hover:bg-white"}`}>{text}</button>
      ))}
    </div>
  );

  // In webcam mode the drop zone is replaced by the camera, until a photo has been captured.
  const webcamView = mode === "webcam" && !source
    ? <Webcam disabled={busy} onCapture={(file) => setSource({ file, name: file.name, preview: URL.createObjectURL(file) })} />
    : undefined;

  return (
    <WorkspaceLayout>
      <PageHeader number={4} title="Face-to-Sketch Generator" description="Turn a face photograph into a sketch in one of three styles." />
      <ErrorBanner message={error} onClose={() => setError(null)} />

      <div className="grid grid-cols-2 gap-7">
        <UploadPanel source={source} onSource={setSource} disabled={busy} onError={setError} sampleKind="faces" tabs={tabs}
          toggle={{ label: "Photo is already a cropped face", checked: cropped, onChange: setCropped }}>
          {webcamView}
        </UploadPanel>
        <Panel title="Style">
          <div className="flex h-full flex-col gap-4">
            <div>
              <div className="label mb-2">Sketch style</div>
              <div className="grid grid-cols-3 gap-1.5 rounded-full border-2 border-edge bg-inset p-1.5">
                {[1, 2, 3].map((n) => (
                  <button key={n} onClick={() => { setStyle(n); setResult(null); }} disabled={busy}
                    className={`h-9 rounded-full text-[13px] font-extrabold transition-colors ${style === n ? "bg-accent text-on-accent" : "text-ink-2 hover:bg-white"}`}>
                    Style {n}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center justify-between rounded-control border-2 border-edge bg-inset px-4 py-2.5">
              <span className="label">Framing</span>
              <span className="num text-[12.5px] text-ink">{cropped ? "used as is (resized)" : "centre square crop"}</span>
            </div>
            <Button variant="primary" className="mt-auto" onClick={generate} disabled={!source || busy}>
              {busy ? "Generating…" : "Generate sketch"}
            </Button>
          </div>
        </Panel>
      </div>

      <div className="mt-9 grid grid-cols-2 gap-7">
        <ImagePanel label="Photo" src={result?.photo_image ?? source?.preview} empty="Awaiting photo" icon={<ImageIcon width={22} height={22} />} />
        <ImagePanel label="Sketch" src={result?.sketch_image} busy={busy}
          empty={busy ? "Running generator" : "Awaiting generation"} icon={<WandIcon width={22} height={22} />} />
      </div>

      {busy && <ProgressStatus startedAt={startedAt} label="Running generator" />}
      {result && !busy && (
        <ResultStrip downloadUrl={result.sketch_image} downloadName={`sketch_style${result.style}.png`} downloadLabel="Download sketch"
          items={[["Style", `Style ${result.style}`], ["Framing", cropped ? "As is" : "Centre crop"], ["Inference time", `${result.timing_ms.inference.toFixed(1)} ms`]]} />
      )}
    </WorkspaceLayout>
  );
}
