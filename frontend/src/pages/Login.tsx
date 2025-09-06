import React from "react";

interface LoginProps {
  onLogin: () => void;
}

const Login: React.FC<LoginProps> = ({ onLogin }) => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300">
      <div className="bg-white shadow-2xl rounded-2xl p-12 w-full max-w-md text-center animate-fadeIn">
        <h1 className="text-3xl font-bold mb-6 text-gray-800">Login</h1>
        <button
          onClick={onLogin}
          className="w-full bg-gradient-to-r from-green-400 to-blue-500 text-white font-semibold py-3 rounded-lg shadow hover:scale-105 transition transform duration-300"
        >
          Login
        </button>
      </div>
    </div>
  );
};

export default Login;
