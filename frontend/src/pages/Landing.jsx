// Landing page: one centered welcome screen (no top bar, no cards, no footer).
// Just a greeting, the app name, one line about it, and the link into the workspaces.
import { Link } from "react-router-dom";
import { ArrowRightIcon } from "../components/Icons.jsx";

export default function Landing() {
  return (
    <div className="bg-grid relative flex h-screen flex-col items-center justify-center overflow-hidden px-6 text-center">
      {/* flat decorative dots (purely visual) */}
      <span className="absolute left-[12%] top-[18%] h-16 w-16 rounded-full border-2 border-edge bg-butter" aria-hidden />
      <span className="absolute right-[14%] top-[24%] h-10 w-10 rounded-full border-2 border-edge bg-mint" aria-hidden />
      <span className="absolute bottom-[20%] left-[18%] h-12 w-12 rounded-full border-2 border-edge bg-lilac" aria-hidden />
      <span className="absolute bottom-[16%] right-[12%] h-20 w-20 rounded-full border-2 border-edge bg-soft" aria-hidden />

      <span className="rounded-full border-2 border-edge bg-white px-5 py-1.5 text-[clamp(14px,2vh,18px)] font-extrabold text-ink shadow-pop-sm">
        Hello, welcome to
      </span>

      <h1 className="mt-[3vh] font-serif text-[clamp(88px,22vh,220px)] leading-[0.95] font-semibold italic text-accent">
        Bloom
      </h1>

      <p className="mt-[3vh] max-w-[520px] text-[clamp(16px,2.2vh,20px)] font-bold text-ink-2">
        Bring damaged photos back to life and turn faces into sketches.
      </p>

      <Link to="/universal"
        className="mt-[5vh] inline-flex h-[clamp(50px,7vh,62px)] items-center gap-4 rounded-full border-2 border-edge bg-accent px-10 text-[clamp(16px,2.1vh,18px)] font-extrabold text-on-accent shadow-pop transition-all hover:bg-accent-strong active:translate-x-[3px] active:translate-y-[3px] active:shadow-none">
        Enter workspace <ArrowRightIcon width={20} height={20} />
      </Link>
    </div>
  );
}