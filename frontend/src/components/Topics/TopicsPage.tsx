import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Pie } from "react-chartjs-2";
import ChartDataLabels from "chartjs-plugin-datalabels";
import ReactMarkdown from "react-markdown";
import SummaryPopup from "../SummaryPopup/SummaryPopup";
import "./TopicsPage.css";
import { API_BASE_URL } from "../config";

ChartJS.register(ArcElement, Tooltip, Legend, ChartDataLabels);

const TopicsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const { topics, originalText, keywords } = location.state || {};

  const [selectedSummary, setSelectedSummary] = useState<string | null>(null);
  const [hoveredTopic, setHoveredTopic] = useState<string | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [showTopicPopup, setShowTopicPopup] = useState<boolean>(false);
  const [showStylePopup, setShowStylePopup] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [topicPerspectives, setTopicPerspectives] = useState<{ [topic: string]: any }>({});
  const [selectedTopicPerspectives, setSelectedTopicPerspectives] = useState<any | null>(null);
  const [showSummaryPopup, setShowSummaryPopup] = useState<boolean>(false);
  const [mergedSummaries, setMergedSummaries] = useState<{ [topic: string]: string }>({});
  const [loadingPerspectives, setLoadingPerspectives] = useState<boolean>(false);
  const [perspectives, setPerspectives] = useState<{ [key: string]: string }>({});


  const styles = [
    "formal",
    "academic",
    "gen_z",
    "narrative",
    "persuasive",
    "satirical",
    "conversational",
    "poetic",
    "investigative",
  ];

  useEffect(() => {
    const fetchPerspectives = async () => {
      setLoadingPerspectives(true);
      try {
        const response = await fetch(`${API_BASE_URL}/topics/Mistral/perspectives`);
        if (response.ok) {
          const data = await response.json();
          const perspectiveData = data.perspectives || data;
          setTopicPerspectives(perspectiveData);
        } else {
          console.error("Failed to fetch perspectives.");
        }
      } catch (error) {
        console.error("Error fetching perspectives:", error);
      } finally {
        setLoadingPerspectives(false);
      }
    };
    
    fetchPerspectives();
  }, []);

  const handleBubbleClick = (topicName: string) => {
    setSelectedTopic(topicName);
    setSelectedTopicPerspectives(topicPerspectives[topicName]);
    setSelectedSummary(mergedSummaries[topicName] || null);
  };

  const handleMergedSummarySave = (topic: string, summary: string) => {
    setMergedSummaries(prev => ({ ...prev, [topic]: summary }));
    setSelectedSummary(summary);
    setShowSummaryPopup(false);
  };

  const chartData = {
    labels: Object.keys(topics),
    datasets: [
      {
        data: Object.values(topics),
        backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"],
        hoverBackgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"],
      },
    ],
  };

  const handleTopicClick = (topic: string) => {
    setSelectedTopic(topic);
    setShowTopicPopup(false);
    setShowStylePopup(true);
  };

  const handleStyleClick = async (style: string) => {
    setIsLoading(true);
    if (selectedTopic && mergedSummaries[selectedTopic]) {
      try {
        const requestBody = {
          topic: selectedTopic,
          summary: mergedSummaries[selectedTopic],
          keywordsSent: keywords[selectedTopic],
          style,
        };

        const response = await fetch(`${API_BASE_URL}/generate-news`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });

        if (response.ok) {
          const { article } = await response.json();
          // ✅ Send additional context to NewsArticlePage
          navigate("/news-article", {
            state: {
              article,
              topic: selectedTopic,
              originalText,
              mergedSummary: mergedSummaries[selectedTopic],
              perspectives: topicPerspectives[selectedTopic],
              allTopics: topics,
              topicProbabilities: Object.values(topics),
              allPerspectives: topicPerspectives,
            }
        });
        } else {
          console.error("Failed to generate news article.");
        }
      } catch (error) {
        console.error("Error generating news article:", error);
      } finally {
        setIsLoading(false);
      }
    }
  };

  if (!topics || !originalText) {
    return (
      <div className="topics-page">
        <h2>No topics to display</h2>
        <button onClick={() => navigate("/")}>Go Back</button>
      </div>
    );
  }

  return (
    <div className="topics-container">
      <div className="topics-left">
        <h1>Extracted Topics</h1>

        <div className="original-text-container">
          <h2>Original Text:</h2>
          <div className="scrollable-text">
            <p>{originalText}</p>
          </div>
        </div>

        <div className="topics-list-container">
          <div className="division">
            <div className="subcolumn">
              <h2>Suggested Topics</h2>
              <p>Click on Each Topic to see Perspectives & Summarize</p>
              <div className="topics-bubbles">
                {Object.entries(topics).map(([topicName, percentage], index) => (
                  <div
                    key={index}
                    className="bubble"
                    data-percentage={`${percentage}%`}
                    onClick={() => handleBubbleClick(topicName)}
                    onMouseEnter={() => setHoveredTopic(`${topicName} (${percentage}%)`)}
                    onMouseLeave={() => setHoveredTopic(null)}
                  >
                    <span className="bubble-text">{topicName}</span>
                  </div>
                ))}
                {hoveredTopic && <div className="tooltip">{hoveredTopic}</div>}
              </div>
            </div>
            <div className="subcolumn">
              <div className="chart-container">
                <h2>Topic Distribution</h2>
                <Pie data={chartData} />
              </div>
            </div>
          </div>
        </div>

        <button className="back-button" onClick={() => navigate("/")}>Back to Home</button>
        <button className="generate-article-button" onClick={() => setShowTopicPopup(true)}>
          Generate News Article
        </button>
      </div>

      <div className="topics-right">
        <h1>Summary</h1>
        <div className="summary-container">
          <div className="scrollable-text-summary">
            {selectedSummary ? (
              <ReactMarkdown>{selectedSummary}</ReactMarkdown>
            ) : (
              <div className="summary-placeholder">
                <p>Click Summarize button to check Perspective Aware Summaries and then merge them to view summary here.</p>
              </div>
            )}
          </div>
          {selectedTopic && (
            <button onClick={() => setShowSummaryPopup(true)}>Summarize</button>
          )}
        </div>

        <h2>Multiple Perspectives</h2>
        {selectedTopicPerspectives ? (
          <>
            <p>
              <strong>Agree:</strong> {selectedTopicPerspectives.Agree}% &nbsp; | &nbsp;
              <strong>Disagree:</strong> {selectedTopicPerspectives.Disagree}%
            </p>
            <div className="perspectives-container scrollable-text-summary">
              {Object.entries(selectedTopicPerspectives.Perspectives).map(([key, val]: [string, any]) => (
                <details key={key} className="perspective-item">
                  <summary>{val.Stance}</summary>
                  <p><strong>Reason:</strong> {val.Reason}</p>
                  <p><strong>Criteria:</strong> {val.Criteria.join(", ")}</p>
                </details>
              ))}
            </div>
          </>
        ) : (
          <p>Select a topic to view perspectives.</p>
        )}
      </div>

      {showTopicPopup && (
        <div className="popup-overlay">
          <div className="popup-content">
            <h2>To generate News Article, choose the topic from below:</h2>
            <ul>
              {Object.keys(topics).map((topic, index) => (
                <li key={index} onClick={() => handleTopicClick(topic)}>
                  {topic}
                </li>
              ))}
            </ul>
            <button onClick={() => setShowTopicPopup(false)}>Close</button>
          </div>
        </div>
      )}

      {showStylePopup && (
        <div className="popup-overlay">
          {isLoading ? (
            <div className="loading-screen">
              <div className="spinner"></div>
              <p>Loading, please wait...</p>
            </div>
          ) : (
            <div className="popup-content">
              <h2>Choose a style</h2>
              <ul>
                {styles.map((style, index) => (
                  <li key={index} onClick={() => handleStyleClick(style)}>
                    {style}
                  </li>
                ))}
              </ul>
              <button onClick={() => setShowStylePopup(false)}>Close</button>
            </div>
          )}
        </div>
      )}

      {showSummaryPopup && selectedTopic && (
        <SummaryPopup
          topic={selectedTopic}
          onClose={() => setShowSummaryPopup(false)}
          onSaveMergedSummary={handleMergedSummarySave}
        />
      )}
    </div>
  );
};

export default TopicsPage;
