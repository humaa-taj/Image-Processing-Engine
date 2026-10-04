// Every network call of the app lives here. The contract is documented in docs/api_contract.md.
//
// Mock mode (VITE_MOCK=1) returns placeholder responses with the same shapes, for UI work without
// a backend. It is OFF by default, and the top bar shows a "MOCK DATA" badge whenever it is on.
// Mock output is never real and must never be used in the report.

export const MOCK = import.meta.env.VITE_MOCK === "1";
export const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL || "http://127.0.0.1:5000";

/** Error with a message that can be shown to the user as is. */
export class ApiError extends Error {}

async function request(path, options) {
  let res;
  try {
    res = await fetch(`/api${path}`, options);
  } catch {
    throw new ApiError("Backend unreachable. Is the backend running?");
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error page (e.g. proxy error) */
  }
  if (!res.ok) {
    const detail = body?.detail;
    const text = typeof detail === "string" ? detail : res.status >= 500 ? "Backend unreachable or failed." : "Request failed.";
    throw new ApiError(`${text} (HTTP ${res.status})`);
  }
  return body;
}

/** multipart form with the image (uploaded file OR sample name) + extra fields. */
function form(source, fields) {
  const fd = new FormData();
  if (source.file) fd.append("file", source.file);
  else fd.append("sample", source.sample);
  for (const [k, v] of Object.entries(fields)) if (v !== undefined && v !== null) fd.append(k, String(v));
  return fd;
}

/** Fields shared by the restoration endpoints. */
function restoreFields({ corruption, level, seed, alreadyCorrupted }) {
  const type = alreadyCorrupted ? "none" : corruption;
  return { corruption: type, level, seed, input_is_clean: !alreadyCorrupted && type === "none" };
}

export const api = {
  health: () => (MOCK ? mock.health() : request("/health")),
  samples: () => (MOCK ? mock.samples() : request("/samples")),
  corrupt: (source, opts) =>
    MOCK ? mock.corrupt(opts) : request("/corrupt", { method: "POST", body: form(source, { corruption: opts.corruption, level: opts.level, seed: opts.seed }) }),
  restoreUniversal: (source, opts) =>
    MOCK ? mock.universal(opts) : request("/restore/universal", { method: "POST", body: form(source, restoreFields(opts)) }),
  restoreHard: (source, opts) =>
    MOCK ? mock.hard(opts) : request("/restore/hard", { method: "POST", body: form(source, restoreFields(opts)) }),
  restoreSoft: (source, opts) =>
    MOCK ? mock.soft(opts) : request("/restore/soft", { method: "POST", body: form(source, restoreFields(opts)) }),
  /** style: 1 | 2 | 3; fit: "crop" (centre square crop) | "stretch" (photo is already a cropped face). */
  sketch: (source, { style, fit }) =>
    MOCK ? mock.sketch(style) : request("/sketch", { method: "POST", body: form(source, { style, fit }) }),
};

// ------------------------------------------------------------------ mock mode (placeholders only)

/** A flat grey placeholder image labelled MOCK, so it can never be mistaken for a model output. */
function placeholder(label) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = "#2a2b2e";
  g.fillRect(0, 0, 128, 128);
  g.fillStyle = "#979ba1";
  g.font = "11px monospace";
  g.textAlign = "center";
  g.fillText("MOCK", 64, 60);
  g.fillText(label, 64, 76);
  return c.toDataURL("image/png");
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const mockCorruption = (o) =>
  o.alreadyCorrupted || o.corruption === "none" ? null : { type: o.corruption, level: o.level, seed: 1, severity: 0, params: {} };
const mockRef = (o) => {
  const ok = !o.alreadyCorrupted;
  return { reference_available: ok, clean_image: ok ? placeholder("clean") : null, error_map_image: ok ? placeholder("error map") : null, mean_abs_error: null };
};

const mock = {
  async health() {
    return { status: "ok", models_loaded: 0, models_expected: 7, lfs_pointer_files: [], models: {}, mock: true };
  },
  async samples() {
    const list = (kind) => Array.from({ length: 4 }, (_, i) => ({ name: `${kind}/mock_${i}.png`, url: placeholder(`sample ${i + 1}`) }));
    return { pets: list("pets"), faces: list("faces") };
  },
  async corrupt(o) {
    await wait(300);
    return { corruption: mockCorruption(o), input_image: placeholder("corrupted"), clean_image: placeholder("clean"), timing_ms: { total: 0 } };
  },
  async universal(o) {
    await wait(900);
    return { task: "Universal Restoration", corruption: mockCorruption(o), input_image: placeholder("input"), output_image: placeholder("output"), ...mockRef(o), timing_ms: { inference: 0, total: 0 } };
  },
  async hard(o) {
    await wait(900);
    return {
      task: "Hard-Routed Restoration", corruption: mockCorruption(o),
      probabilities: { clean: 0.25, salt: 0.25, blur: 0.25, occlusion: 0.25 }, predicted_class: "clean", selected_expert: "identity",
      input_image: placeholder("input"), output_image: placeholder("output"), ...mockRef(o), timing_ms: { classifier: 0, expert: 0, inference: 0, total: 0 },
    };
  },
  async soft(o) {
    await wait(900);
    const weights = { identity: 0.25, "salt expert": 0.25, "blur expert": 0.25, "occlusion expert": 0.25 };
    return {
      task: "Soft Mixture-of-Experts Restoration", corruption: mockCorruption(o), weights,
      ranking: Object.keys(weights), dominant_branch: "identity", top_contributors: Object.keys(weights), top_contributor_threshold: 0.1,
      input_image: placeholder("input"), output_image: placeholder("output"), ...mockRef(o), timing_ms: { inference: 0, total: 0 },
    };
  },
  async sketch(style) {
    await wait(900);
    return { task: "Face-to-Sketch Generator", style, photo_image: placeholder("photo"), sketch_image: placeholder(`sketch style ${style}`), timing_ms: { inference: 0, total: 0 } };
  },
};
