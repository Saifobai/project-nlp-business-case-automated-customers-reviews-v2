import { Link } from "react-router-dom";
import { Film, Info } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="fixed top-0 inset-x-0 z-50 bg-[#0a0a0a]/80 backdrop-blur border-b border-yellow-400/20 shadow-[0_0_25px_rgba(250,204,21,0.15)]">
      <div className="mx-auto max-w-7xl px-6 py-3 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 group">
          <Film className="h-7 w-7 text-yellow-400 drop-shadow-[0_0_12px_rgba(250,204,21,0.6)]" />
          <span className="text-2xl font-extrabold bg-gradient-to-r from-yellow-300 to-yellow-500 bg-clip-text text-transparent group-hover:from-yellow-200 group-hover:to-yellow-400 transition">
            IMDB Insights
          </span>
        </Link>

        <div className="flex items-center gap-8 text-sm">
          <Link
            to="/"
            className="text-zinc-300 hover:text-yellow-400 transition"
          >
            Dashboard
          </Link>
          <Link
            to="/about"
            className="text-zinc-300 hover:text-yellow-400 transition flex items-center gap-1"
          >
            <Info className="h-4 w-4" /> About
          </Link>
        </div>
      </div>
    </nav>
  );
}
