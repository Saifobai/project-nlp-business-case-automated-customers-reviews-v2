import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const SENTIMENT_COLORS = ["#22d3ee", "#facc15", "#ef4444"];

export default function SentimentChart({ data }) {
  return (
    <div className="bg-[#0d0f16]/80 border border-yellow-400/20 rounded-2xl p-4">
      <h2 className="text-lg font-bold text-yellow-400 mb-2">
        Sentiment Distribution
      </h2>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              outerRadius={90}
              label
            >
              {data.map((_, i) => (
                <Cell
                  key={i}
                  fill={SENTIMENT_COLORS[i % SENTIMENT_COLORS.length]}
                />
              ))}
            </Pie>
            <Tooltip
              wrapperStyle={{
                background: "#0b0e17",
                border: "1px solid #3f3f46",
                borderRadius: 8,
              }}
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
