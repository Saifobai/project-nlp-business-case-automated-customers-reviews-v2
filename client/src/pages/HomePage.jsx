import { useEffect, useState } from "react";
import axios from "axios";
import CountUp from "react-countup";

// Components
import Sidebar from "../components/Sidebar";
import StatCard from "../components/StatCard";
import AddReviewForm from "../components/AddReviewForm";
import SearchMovies from "../components/SearchMovies";
import WordCloud from "../components/WordCloud";
import SentimentChart from "../components/SentimentChart";
import GenreExplorer from "../components/GenreExplorer";
import ReviewsList from "../components/ReviewsList";
import AISummary from "../components/AISummary";

// Icons
import { Film, Star, TrendingUp, MessageSquare } from "lucide-react";

export default function HomePage() {
  const [stats, setStats] = useState({});
  const [sentimentData, setSentimentData] = useState([]);
  const [genreBars, setGenreBars] = useState([]);
  const [wordCloudData, setWordCloudData] = useState([]);
  const [reviews, setReviews] = useState([]);
  const [summary, setSummary] = useState("");

  const fetchDashboard = async () => {
    try {
      const res = await axios.get("http://localhost:5000/dashboard");
      setStats(res.data.stats);
      setGenreBars(res.data.genres);
      setWordCloudData(res.data.wordcloud);
      setReviews(res.data.reviews);
      setSentimentData(res.data.sentiment);
    } catch (err) {
      console.error("Error fetching dashboard:", err);
    }
  };

  useEffect(() => {
    fetchDashboard();
  }, []);

  return (
    <div className="flex min-h-screen bg-[#07080c] text-white">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content */}
      <main className="flex-1 p-8 overflow-y-auto mt-9">
        {/* Add Review & Search Section */}
        <section className="bg-[#0d0f16]/80 border border-yellow-400/20 rounded-2xl p-6 mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <AddReviewForm
              setSummary={setSummary}
              refresh={fetchDashboard}
              setSentimentData={setSentimentData}
            />
            <SearchMovies />
          </div>
        </section>

        {/* Dashboard Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Stats + Word Cloud */}
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <StatCard
                icon={Film}
                label="Total Reviews"
                value={
                  <CountUp
                    end={stats.totalReviews || 0}
                    duration={1.2}
                    separator=","
                  />
                }
              />
              <StatCard
                icon={Star}
                label="Avg Rating"
                value={`${stats.avgRating || 0}/10`}
              />
              <StatCard
                icon={TrendingUp}
                label="Top Genre"
                value={stats.topGenre || "N/A"}
                glow="violet"
              />
              <StatCard
                icon={MessageSquare}
                label="Positive Share"
                value={`${stats.positiveShare || 0}%`}
                glow="cyan"
              />
            </div>
            <WordCloud data={wordCloudData} />
          </div>

          {/* Middle Column: Charts */}
          <div className="space-y-6">
            <SentimentChart data={sentimentData} />
            <GenreExplorer data={genreBars} />
          </div>

          {/* Right Column: Reviews + Summary */}
          <div className="space-y-6">
            <ReviewsList reviews={reviews} />
            <AISummary summary={summary} />
          </div>
        </div>
      </main>
    </div>
  );
}
