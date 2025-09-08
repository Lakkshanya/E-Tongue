import React from "react";
import "./Results.css";

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
    <div className="results-card">
      <h2 className="results-title">Latest Prediction</h2>
      
      <div className="results-grid">
        <div className="result-item">
          <div className="result-icon">
            {prediction.adulterated ? (
              <span className="icon-cross">✅</span>
            ) : (
              <span className="icon-check">❌</span>
            )}
          </div>
          <h3 className="result-label">Adulteration Status</h3>
          <p className="result-value">
            {prediction.adulterated ? "Adulterated" : "Pure"}
          </p>
          <div className="result-description">
            {prediction.adulterated 
              ? "Substance contains impurities" 
              : "No adulterants detected"}
          </div>
        </div>
        
        <div className="result-item">
          <div className="result-icon">
            {prediction.safe_to_use ? (
              <span className="icon-check">✅</span>
            ) : (
              <span className="icon-cross">❌</span>
            )}
          </div>
          <h3 className="result-label">Suitability for Medicine</h3>
          <p className="result-value">
            {prediction.safe_to_use ? "Suitable" : "Not Suitable"}
          </p>
          <div className="result-description">
            {prediction.safe_to_use 
              ? "Safe for medicinal use" 
              : "Not recommended for medicinal use"}
          </div>
        </div>
        
        <div className="result-item">
          <div className="result-icon confidence">
            <span className="confidence-value">{Math.round(prediction.confidence * 100)}%</span>
          </div>
          <h3 className="result-label">Confidence Level</h3>
          <p className="result-value">
            {prediction.confidence > 0.8 ? "High" : 
             prediction.confidence > 0.6 ? "Medium" : "Low"}
          </p>
          <div className="confidence-bar">
            <div 
              className="confidence-fill"
              style={{ width: `${prediction.confidence * 100}%` }}
            ></div>
          </div>
        </div>
      </div>
      
      <div className="result-footer">
        <p className="timestamp">
          Analyzed: {new Date(prediction.createdAt).toLocaleString()}
        </p>
      </div>
    </div>
  );
};

export default Results;