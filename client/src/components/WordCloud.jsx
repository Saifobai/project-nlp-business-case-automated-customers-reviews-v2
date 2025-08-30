export default function WordCloud({ data }) {
  return (
    <div className="bg-[#0d0f16]/80 border border-yellow-400/20 rounded-2xl p-4">
      <h2 className="text-lg font-bold text-yellow-400 mb-2">Word Cloud</h2>
      <div className="flex flex-wrap gap-3">
        {data.map((w, i) => (
          <span
            key={i}
            className="text-yellow-300"
            style={{
              fontSize: `${w.size}px`,
              textShadow: "0 0 10px rgba(250,204,21,0.4)",
            }}
          >
            {w.word}
          </span>
        ))}
      </div>
    </div>
  );
}
