import { useState } from "react";
import axios from "axios";
import { Search } from "lucide-react";

export default function AddReviewForm({
  setSummary,
  refresh,
  setSentimentData,
}) {
  const [movieName, setMovieName] = useState("");
  const [customReview, setCustomReview] = useState("");
  const [reviewGenre, setReviewGenre] = useState("Drama");
  const [rating, setRating] = useState(5);

  const handleAnalyzeReview = async (e) => {
    e.preventDefault();
    try {
      if (customReview) {
        const res = await axios.post("http://localhost:5000/reviews", {
          movie: movieName,
          text: customReview,
          genre: reviewGenre,
          rating: rating,
        });

        setSummary(res.data.summary || "");

        if (Array.isArray(res.data.sentiment)) {
          setSentimentData(
            res.data.sentiment.map((s) => ({
              name: s.label,
              value: Math.round(s.score * 100),
            }))
          );
        }

        refresh();
        setMovieName("");
        setCustomReview("");
        setRating(5);
      }
    } catch (err) {
      console.error("Error calling API:", err);
    }
  };

  return (
    <form onSubmit={handleAnalyzeReview} className="flex flex-col gap-4">
      <h2 className="text-lg font-bold text-yellow-400 mb-2 flex items-center gap-2">
        <Search className="h-5 w-5" /> Add Review
      </h2>

      <input
        type="text"
        placeholder="Enter your choice of movie..."
        value={movieName}
        onChange={(e) => setMovieName(e.target.value)}
        className="px-4 py-2 rounded-lg bg-[#0b0e17] border border-zinc-700 text-zinc-300 focus:ring-2 focus:ring-yellow-400"
      />
      <textarea
        placeholder="Paste a review..."
        value={customReview}
        onChange={(e) => setCustomReview(e.target.value)}
        className="px-4 py-2 rounded-lg bg-[#0b0e17] border border-zinc-700 text-zinc-300 h-24 focus:ring-2 focus:ring-yellow-400"
      />

      <div className="flex gap-4 flex-col md:flex-row">
        <select
          value={reviewGenre}
          onChange={(e) => setReviewGenre(e.target.value)}
          className="flex-1 px-3 py-2 rounded-lg bg-[#0b0e17] border border-zinc-700 text-zinc-300"
        >
          <option>Drama</option>
          <option>Action</option>
          <option>Comedy</option>
          <option>Romance</option>
          <option>Sci-Fi</option>
          <option>Horror</option>
          <option>Thriller</option>
          <option>Musical</option>
        </select>

        <input
          type="number"
          min="1"
          max="10"
          value={rating}
          onChange={(e) => setRating(Number(e.target.value))}
          className="w-28 px-3 py-2 rounded-lg bg-[#0b0e17] border border-zinc-700 text-zinc-300"
        />

        <button
          type="submit"
          className="px-6 py-2 rounded-lg bg-yellow-400 text-black font-bold hover:bg-yellow-300 transition"
        >
          Analyze & Add
        </button>
      </div>
    </form>
  );
}
