// Small inline SVG icons (no icon library needed). Stroke = currentColor, so they follow the text colour.
const base = { width: 16, height: 16, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2.2, strokeLinecap: "round", strokeLinejoin: "round" };

export const UploadIcon = (p) => (<svg {...base} {...p}><path d="M12 16V4M7 9l5-5 5 5M5 20h14" /></svg>);
export const ArrowRightIcon = (p) => (<svg {...base} {...p}><path d="M5 12h14M13 6l6 6-6 6" /></svg>);
export const ImageIcon = (p) => (<svg {...base} {...p}><rect x="3" y="4" width="18" height="16" rx="4" /><circle cx="9" cy="10" r="1.6" /><path d="M21 16l-5-5-9 9" /></svg>);
export const FrameIcon = (p) => (<svg {...base} {...p}><path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" /></svg>);
export const WandIcon = (p) => (<svg {...base} {...p}><path d="M4 20L16 8M14 4v2M18 8h2M17 4.5l1.5-1.5M19.5 7L21 5.5" /></svg>);
export const DownloadIcon = (p) => (<svg {...base} {...p}><path d="M12 4v12M7 11l5 5 5-5M5 20h14" /></svg>);
export const ChevronIcon = (p) => (<svg {...base} {...p}><path d="M8 10l4 4 4-4" /></svg>);
export const SamplesIcon = (p) => (<svg {...base} {...p}><rect x="3" y="3" width="8" height="8" rx="3" /><rect x="13" y="3" width="8" height="8" rx="3" /><rect x="3" y="13" width="8" height="8" rx="3" /><rect x="13" y="13" width="8" height="8" rx="3" /></svg>);
export const CloseIcon = (p) => (<svg {...base} {...p}><path d="M6 6l12 12M18 6L6 18" /></svg>);
