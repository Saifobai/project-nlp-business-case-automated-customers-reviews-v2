import { motion } from "framer-motion";
import { useMemo } from "react";

export default function StatCard({
  icon: Icon,
  label,
  value,
  glow = "yellow",
}) {
  const shadow = useMemo(() => {
    if (glow === "cyan") return "shadow-[0_0_22px_rgba(34,211,238,0.35)]";
    if (glow === "violet") return "shadow-[0_0_22px_rgba(167,139,250,0.35)]";
    return "shadow-[0_0_22px_rgba(250,204,21,0.35)]";
  }, [glow]);

  return (
    <motion.div
      whileHover={{ scale: 1.04 }}
      className={`bg-[#0d0f16]/80 border border-yellow-400/20 rounded-2xl p-4 ${shadow}`}
    >
      <div className="flex items-center gap-2">
        <Icon className="h-5 w-5 text-yellow-400 drop-shadow-[0_0_10px_rgba(250,204,21,0.6)]" />
        <span className="text-zinc-300 text-sm">{label}</span>
      </div>
      <div className="mt-1 text-2xl font-extrabold text-yellow-400">
        {value}
      </div>
    </motion.div>
  );
}
