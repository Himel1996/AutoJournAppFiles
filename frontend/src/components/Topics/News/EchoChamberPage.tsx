import React, { useEffect, useState, useRef } from "react";
import { useLocation } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import axios from "axios";
import "./EchoChamberPage.css";
import RadarStanceChart from "./RadarStanceChart";

type StancePair = {
  stance_pair: [string, string];
  description: string;
};

type ClaimVerification = {
  claim: string;
  label: string;      // GPT-4 output: "true", "half true", etc.
  summary: string;    // Evidence summary
};

const labelToColor = (label: string) => {
  switch (label.toLowerCase()) {
    case "true":
      return "green";
    case "mostly true":
      return "#4caf50";
    case "half true":
      return "#f57c00";
    case "barely true":
      return "#ff9800";
    case "false":
      return "red";
    case "pants-on-fire":
      return "darkred";
    default:
      return "#555";
  }
};

const EchoChamberPage = () => {
  const location = useLocation();
  const {
    article,
    graphData,
    originalText,
    topicName,
    topicProbabilities,
    allPerspectives,
    allTopics,
    summary,
    biasStats,
  } = location.state || {};

  const [stancePairs, setStancePairs] = useState<StancePair[]>([]);
  const [adjustedArticle, setAdjustedArticle] = useState(article);
  const [sliderValues, setSliderValues] = useState<number[]>([]);
  const [verifications, setVerifications] = useState<ClaimVerification[]>([]);
  const [loadingVerification, setLoadingVerification] = useState(false);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const fetchExtremes = async () => {
      try {
        const res = await axios.post("http://72.62.44.22:8000/detect-echo-extremes", {
          originalText: originalText,
        });
        setStancePairs(res.data.extremes);
      } catch (err) {
        console.error("Error fetching echo chamber extremes", err);
      }
    };

    fetchExtremes();
  }, [originalText]);

  useEffect(() => {
    setSliderValues(new Array(stancePairs.length).fill(50));
  }, [stancePairs]);

  const handleSliderChange = async (index: number, value: number) => {
    const updatedValues = [...sliderValues];
    updatedValues[index] = value;
    setSliderValues(updatedValues);

    const stanceData = {
      original_article: article,
      stance_pair: stancePairs[index].stance_pair,
      slider_value: value,
    };

    try {
      const res = await axios.post("http://72.62.44.22:8000/shift-stance", stanceData);
      setAdjustedArticle(res.data.shifted_article);
    } catch (err) {
      console.error("Error shifting stance:", err);
    }
  };

  const debouncedSliderChange = (index: number, value: number) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => handleSliderChange(index, value), 400);
  };

  const handleOpenClaimVerificationPage = () => {
    const pageState = {
      article: adjustedArticle,
    };
    const url = "/claim-verification";
    const newWindow = window.open(url, "_blank");
    if (newWindow) {
      newWindow.name = JSON.stringify(pageState);
    }
  };

  return (
    <div className="echo-chamber-container">
      <div className="left-panel" style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <div style={{ flex: 1, overflowY: "auto", padding: "10px" }}>
          {stancePairs.length > 0 && (
            <>
              <h3>Adjust Stance Perspective</h3>
              {stancePairs.map((pair, index) => (
                <div key={index} className="stance-slider-block" style={{ marginBottom: "16px" }}>
                  <label style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>🟥 {pair.stance_pair[1]}</span>
                    <span>🟦 {pair.stance_pair[0]}</span>
                  </label>

                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={sliderValues[index]}
                    onChange={(e) => debouncedSliderChange(index, parseInt(e.target.value))}
                    style={{ width: "100%" }}
                  />

                  <p style={{ fontSize: "0.85em", color: "#555" }}>{pair.description}</p>
                </div>
              ))}
            </>
          )}

          {stancePairs.length > 0 && sliderValues.length > 0 && (
            <div
              style={{
                marginBottom: "20px",
                padding: "10px",
                backgroundColor: "#f9f9f9",
                borderRadius: "8px",
              }}
            >
              <h3 style={{ marginBottom: "10px" }}>🧭 Stance Shift Overview</h3>
              <RadarStanceChart stancePairs={stancePairs} sliderValues={sliderValues} />
            </div>
          )}
        </div>
      </div>

      <div className="right-panel">
        <h2>Generated News Article</h2>
        <div className="article-scroll">
          <ReactMarkdown>{adjustedArticle}</ReactMarkdown>
        </div>

        <button
          onClick={handleOpenClaimVerificationPage}
          style={{
            marginTop: "20px",
            padding: "10px 15px",
            backgroundColor: "#1976d2",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          🔍 Verify Factuality
        </button>


        {verifications.length > 0 && (
          <div style={{ marginTop: "20px" }}>
            <h3>🧾 Claim Verification Results</h3>
            <ul>
              {verifications.map((v, idx) => (
                <li
                  key={idx}
                  style={{
                    marginBottom: "16px",
                    border: "1px solid #ddd",
                    borderRadius: "8px",
                    padding: "12px",
                    backgroundColor: "#fafafa",
                  }}
                >
                  <p>
                    <strong>🔍 Claim:</strong> {v.claim}
                  </p>
                  <p>
                    <strong>🧠 Verdict:</strong>{" "}
                    <span style={{ fontWeight: "bold", color: labelToColor(v.label) }}>
                      {v.label.toUpperCase()}
                    </span>
                  </p>
                  <p>
                    <strong>📄 Summary of Evidence:</strong> {v.summary}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default EchoChamberPage;
