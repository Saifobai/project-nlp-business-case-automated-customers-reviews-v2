import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

const GENRE_BAR = "#a78bfa";

export default function GenreExplorer({ data }) {
  const [selectedGenre, setSelectedGenre] = useState("Drama");

  return (
    <div className="bg-[#0d0f16]/80 border border-yellow-400/20 rounded-2xl p-4">
      <div className="flex justify-between mb-2">
        <h2 className="text-lg font-bold text-yellow-400">Genre Explorer</h2>
        <select
          value={selectedGenre}
          onChange={(e) => setSelectedGenre(e.target.value)}
          className="bg-[#0b0e17] border border-zinc-700 rounded-lg px-2 py-1 text-sm text-zinc-300"
        >
          {data
            .filter((g) => g.genre !== "Unknown")
            .map((g) => (
              <option key={g.genre} value={g.genre}>
                {g.genre}
              </option>
            ))}
        </select>
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data.filter((g) => g.genre !== "Unknown")}>
            <CartesianGrid stroke="#1f2937" vertical={false} />
            <XAxis dataKey="genre" stroke="#a1a1aa" />
            <YAxis stroke="#a1a1aa" />
            <Tooltip
              wrapperStyle={{
                background: "#0b0e17",
                border: "1px solid #3f3f46",
                borderRadius: 8,
              }}
            />
            <Bar dataKey="count" fill={GENRE_BAR} radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
