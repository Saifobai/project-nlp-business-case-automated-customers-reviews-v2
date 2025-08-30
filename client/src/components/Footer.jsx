import { FaGithub, FaLinkedin, FaTwitter } from "react-icons/fa";

const Footer = () => {
  return (
    <footer className="bg-[#0b1224] border-t border-gray-700 mt-20 py-6">
      <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
        <p className="text-gray-400 text-sm">
          © {new Date().getFullYear()} AutoReviews. All rights reserved.
        </p>

        <div className="flex gap-6 text-xl">
          <a
            href="https://github.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#4fc3f7] hover:text-white"
          >
            <FaGithub />
          </a>
          <a
            href="https://linkedin.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#ba68c8] hover:text-white"
          >
            <FaLinkedin />
          </a>
          <a
            href="https://twitter.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#e91e63] hover:text-white"
          >
            <FaTwitter />
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
