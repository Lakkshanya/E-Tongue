import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface SensorData {
  features: Record<string, number>;
  temperature?: number;
  createdAt: string;
}

interface ChartsProps {
  sensorData: SensorData[];
}

const Charts: React.FC<ChartsProps> = ({ sensorData }) => {
  // Convert sensorData to chart-friendly format
  const chartData = sensorData.map((d) => ({
    time: new Date(d.createdAt).toLocaleTimeString(),
    pH: d.features.pH || 0,
    TDS: d.features.Solids_TDS || 0,
    Temp: d.temperature || 0,
  }));

  return (
    <div className="bg-white shadow-xl rounded-xl p-6 w-full max-w-6xl animate-fadeIn">
      <h2 className="text-2xl font-bold mb-4 text-gray-800 text-center">Sensor Trends</h2>
      <div className="flex flex-col md:flex-row gap-8">
        <div className="flex-1 bg-green-50 p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">pH Levels</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="pH" stroke="#16a34a" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="flex-1 bg-blue-50 p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">TDS Levels</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="TDS" stroke="#1d4ed8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="flex-1 bg-purple-50 p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Temperature</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="Temp" stroke="#7e22ce" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Charts;
