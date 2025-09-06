import React from "react";

interface SensorData {
  features: Record<string, number>;
  temperature?: number;
  createdAt: string;
}

interface PredictionData {
  adulterated: boolean;
  safe_to_use: boolean;
  borderline: boolean;
  confidence: number;
  sensorRef: SensorData;
  createdAt: string;
}

interface ResultsProps {
  prediction: PredictionData;
}

const Results: React.FC<ResultsProps> = ({ prediction }) => {
  return (
    <div className="bg-white shadow-xl rounded-xl p-6 w-full max-w-3xl text-center animate-fadeIn">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Latest Prediction</h2>
      <div className="flex flex-col md:flex-row justify-around gap-6">
        <div className="p-4 rounded-lg bg-green-100 hover:bg-green-200 transition duration-300">
          <h3 className="font-semibold">Adulteration Status</h3>
          <p className="text-xl font-bold">{prediction.adulterated ? "❌" : "✅"}</p>
        </div>
        <div className="p-4 rounded-lg bg-blue-100 hover:bg-blue-200 transition duration-300">
          <h3 className="font-semibold">Suitability for Medicine</h3>
          <p className="text-xl font-bold">{prediction.safe_to_use ? "✅" : "❌"}</p>
        </div>
        <div className="p-4 rounded-lg bg-purple-100 hover:bg-purple-200 transition duration-300">
          <h3 className="font-semibold">Confidence</h3>
          <p className="text-xl font-bold">{prediction.confidence}</p>
        </div>
      </div>
    </div>
  );
};

export default Results;
