// components/SummaryCard.jsx
export default function SummaryCard({ summary }) {
  return (
    <div className="bg-[#0d0f16]/80 border border-yellow-400/20 rounded-2xl p-4">
      <h2 className="text-lg font-bold text-yellow-400 mb-2">AI Summary</h2>
      <div className="text-zinc-300 text-sm leading-relaxed">
        {summary ? (
          <p>{summary}</p>
        ) : (
          <p className="text-zinc-500">
            Submit a review above to generate an AI summary.
          </p>
        )}
      </div>
    </div>
  );
}
