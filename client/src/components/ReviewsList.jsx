import { motion } from "framer-motion";

export default function ReviewsList({ reviews }) {
  return (
    <div className="bg-[#0d0f16]/80 border border-yellow-400/20 rounded-2xl p-4">
      <h2 className="text-lg font-bold text-yellow-400 mb-2">Live Reviews</h2>
      <ul className="space-y-2">
        {reviews.map((r) => (
          <motion.li
            key={r.id}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
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
          </motion.li>
        ))}
      </ul>
    </div>
  );
}
