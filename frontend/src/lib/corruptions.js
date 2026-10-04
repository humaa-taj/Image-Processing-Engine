// The corruption choices shown in the UI. Severity values are EXACTLY the assignment's test levels
// (src/config.py). The backend applies them with src/corruptions.py; the UI only displays them.
export const CORRUPTIONS = [
  { id: "none", label: "None" },
  { id: "salt", label: "Salt-and-pepper" },
  { id: "blur", label: "Gaussian blur" },
  { id: "occlusion", label: "Rectangular occlusion" },
];

export const LEVELS = [
  { id: "low", label: "Low" },
  { id: "medium", label: "Medium" },
  { id: "high", label: "High" },
];

const PARAMS = {
  salt: { low: "p = 0.03", medium: "p = 0.08", high: "p = 0.15" },
  blur: { low: "kernel 3 / sigma 0.7", medium: "kernel 5 / sigma 1.5", high: "kernel 7 / sigma 2.5" },
  occlusion: { low: "1 rectangle / ~10%", medium: "2 rectangles / ~20%", high: "3 rectangles / ~35%" },
};

/** Text for the Parameters row, e.g. "kernel 5 / sigma 1.5". */
export function paramsText(type, level) {
  return type === "none" ? "—" : PARAMS[type][level];
}

export function corruptionLabel(type) {
  return CORRUPTIONS.find((c) => c.id === type)?.label ?? type;
}

export const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
export const MAX_UPLOAD_MB = 10;

/** Client-side check before uploading (the backend validates again). Returns an error text or null. */
export function validateFile(file) {
  if (!ACCEPTED_TYPES.includes(file.type)) return "Only JPG or PNG images are accepted.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `The file is larger than ${MAX_UPLOAD_MB} MB.`;
  return null;
}
