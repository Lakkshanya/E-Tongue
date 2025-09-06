import React, { useEffect, useState } from "react";
import Charts from "./Charts";
import Results from "./Results";
import axios from "axios";

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

const Dashboard: React.FC = () => {
  const [latestPrediction, setLatestPrediction] = useState<PredictionData | null>(null);
  const [sensorHistory, setSensorHistory] = useState<SensorData[]>([]);

  useEffect(() => {
    fetchLatest();
    fetchHistory();
  }, []);

  const fetchLatest = async () => {
    try {
      const res = await axios.get("http://localhost:4000/api/results?limit=1");
      if (res.data.length) setLatestPrediction(res.data[0]);
    } catch (err) {
      console.error("Error fetching latest prediction:", err);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await axios.get("http://localhost:4000/api/sensors?limit=50");
      setSensorHistory(res.data);
    } catch (err) {
      console.error("Error fetching sensor history:", err);
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-start animate-gradientBG p-6 bg-gradient-to-r from-green-200 via-blue-200 to-purple-200">
      <h1 className="text-5xl font-extrabold mb-8 text-gray-900 text-center drop-shadow-lg">
        Water Quality Dashboard
      </h1>

      {latestPrediction ? (
        <Results prediction={latestPrediction} />
      ) : (
        <p className="text-gray-700 text-lg animate-pulse mt-4">Loading latest prediction...</p>
      )}

      <div className="w-full max-w-6xl mt-10 flex flex-wrap gap-8 justify-center">
        {sensorHistory.length > 0 ? (
          <Charts sensorData={sensorHistory} />
        ) : (
          <p className="text-gray-700 text-lg animate-pulse">Loading sensor history...</p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
