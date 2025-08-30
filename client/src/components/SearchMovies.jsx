import { useState } from "react";
import axios from "axios";

export default function SearchMovies() {
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState([]);

  const handleSearchMovies = async (e) => {
    e.preventDefault();
    const term = searchTerm.trim();
    if (!term) {
      setSearchResults([]);
      return;
    }
    try {
      const res = await axios.get(`http://localhost:5000/search?q=${term}`);
      setSearchResults(res.data);
    } catch (err) {
      console.error("Error searching movies:", err);
      setSearchResults([]);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={handleSearchMovies} className="flex gap-3">
        <input
          type="text"
          placeholder="Search in movies..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="flex-1 px-4 py-2 rounded-lg bg-[#0b0e17] border border-zinc-700 text-zinc-300 focus:ring-2 focus:ring-yellow-400"
        />
        <button
          type="submit"
          className="px-6 py-2 rounded-lg bg-yellow-400 text-black font-bold hover:bg-yellow-300 transition"
        >
          Search
        </button>
      </form>

      <div className="bg-[#0b0e17] border border-yellow-400/20 rounded-xl p-4">
        <h3 className="text-md font-semibold text-yellow-400 mb-2">
          {searchResults.length > 0
            ? `Latest ${searchResults.length} review(s) for “${searchTerm}”`
            : "No search results yet"}
        </h3>
        <ul className="space-y-3 max-h-64 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-yellow-400/40 scrollbar-track-transparent">
          {searchResults.map((r) => (
            <li
              key={r.id}
              className={`rounded-xl px-3 py-2 border text-sm
                ${
                  r.s === "positive"
                    ? "border-cyan-400/30 bg-cyan-400/10"
                    : r.s === "negative"
                    ? "border-red-400/30 bg-red-400/10"
                    : "border-yellow-400/30 bg-yellow-400/10"
                }`}
            >
              <span className="text-yellow-300 font-semibold">{r.movie}:</span>{" "}
              <span className="text-zinc-300">{r.text}</span>
              <div className="text-xs text-zinc-400 mt-1">
                ⭐ {r.rating ?? "—"} | 🎭 {r.genre ?? "—"}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
