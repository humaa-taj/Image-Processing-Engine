import { useState } from "react";
import { api } from "./api.js";

/**
 * State + actions shared by the restoration workspaces (Universal, Hard-Routed, later Soft-MoE).
 * restoreFn = the API call for this workspace, e.g. api.restoreUniversal.
 *
 * Flow: choose a source -> (optional) Apply corruption = backend preview with a seed
 *       -> Restore = backend corrupts again with the SAME seed and runs the model.
 * phase: "idle" | "applying" | "restoring" | "done"
 */
export function useRestoration(restoreFn) {
  const [source, setSourceState] = useState(null);          // { file?, sample?, name, preview }
  const [alreadyCorrupted, setAlreadyCorruptedState] = useState(false);
  const [corruption, setCorruptionState] = useState("none");
  const [level, setLevelState] = useState("medium");
  const [applied, setApplied] = useState(null);              // response of /api/corrupt
  const [result, setResult] = useState(null);                // response of the restore endpoint
  const [phase, setPhase] = useState("idle");
  const [startedAt, setStartedAt] = useState(0);
  const [error, setError] = useState(null);

  // Any change of the inputs makes an old preview / result stale.
  const reset = () => { setApplied(null); setResult(null); setError(null); };
  const setSource = (s) => { setSourceState(s); reset(); };
  const setAlreadyCorrupted = (v) => { setAlreadyCorruptedState(v); reset(); };
  const setCorruption = (v) => { setCorruptionState(v); reset(); };
  const setLevel = (v) => { setLevelState(v); reset(); };

  const busy = phase === "applying" || phase === "restoring";
  const canApply = !!source && !busy && !alreadyCorrupted && corruption !== "none";
  const canRestore = !!source && !busy;

  async function applyCorruption() {
    setPhase("applying"); setError(null); setResult(null); setStartedAt(Date.now());
    try {
      setApplied(await api.corrupt(source, { corruption, level }));
    } catch (e) {
      setError(`Could not apply the corruption: ${e.message}`);
    }
    setPhase("idle");
  }

  async function restore() {
    setPhase("restoring"); setError(null); setStartedAt(Date.now());
    try {
      const seed = applied?.corruption?.seed;   // reuse the previewed corruption exactly
      setResult(await restoreFn(source, { corruption, level, seed, alreadyCorrupted }));
      setPhase("done");
    } catch (e) {
      setError(`Restoration failed: ${e.message}`);
      setPhase("idle");
    }
  }

  /** What the INPUT panel shows: model input after a run, else the preview, else the original. */
  const inputImage = result?.input_image ?? applied?.input_image ?? source?.preview ?? null;

  return {
    source, setSource, alreadyCorrupted, setAlreadyCorrupted, corruption, setCorruption, level, setLevel,
    applied, result, phase, busy, startedAt, error, setError, canApply, canRestore, applyCorruption, restore, inputImage,
  };
}
