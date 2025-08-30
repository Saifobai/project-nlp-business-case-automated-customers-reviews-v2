const AboutPage = () => {
  return (
    <div className="min-h-screen bg-[#0a0f1f] flex items-center justify-center px-6 pt-20">
      <div className="max-w-3xl bg-[#0b1224]/80 border border-[#4fc3f7]/50 rounded-2xl shadow-lg p-10 text-center">
        <h1 className="text-4xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-[#4fc3f7] to-[#2196f3]">
          About AutoReviews
        </h1>
        <p className="text-gray-300 leading-relaxed">
          AutoReviews is a futuristic AI-powered platform that helps businesses
          analyze customer feedback at scale. With our sentiment classifier,
          category explorer, and intelligent summarizer, you can instantly see
          what your customers love, what they struggle with, and how your
          products compare in the market.
        </p>
        <p className="text-gray-400 mt-6">
          Built with React, TailwindCSS, and AI — designed for the future.
        </p>
      </div>
    </div>
  );
};

export default AboutPage;
