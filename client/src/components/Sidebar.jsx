import {
  LayoutDashboard,
  Search,
  BarChart3,
  MessageSquare,
  Cloud,
} from "lucide-react";

export default function Sidebar() {
  return (
    <aside className="w-64 bg-[#0d0f16]/90 border-r border-yellow-400/20 flex flex-col p-6 space-y-8">
      <h1 className="text-2xl font-bold bg-gradient-to-r from-yellow-300 via-yellow-400 to-white bg-clip-text text-transparent">
        🎬 IMDB Insights
      </h1>
      <nav className="flex flex-col gap-4 text-zinc-400">
        <a className="flex items-center gap-2 hover:text-yellow-400 transition cursor-pointer">
          <LayoutDashboard className="w-5 h-5" /> Dashboard
        </a>
        <a className="flex items-center gap-2 hover:text-yellow-400 transition cursor-pointer">
          <Search className="w-5 h-5" /> Search
        </a>
        <a className="flex items-center gap-2 hover:text-yellow-400 transition cursor-pointer">
          <BarChart3 className="w-5 h-5" /> Genres
        </a>
        <a className="flex items-center gap-2 hover:text-yellow-400 transition cursor-pointer">
          <MessageSquare className="w-5 h-5" /> Reviews
        </a>
        <a className="flex items-center gap-2 hover:text-yellow-400 transition cursor-pointer">
          <Cloud className="w-5 h-5" /> Word Cloud
        </a>
      </nav>
    </aside>
  );
}
