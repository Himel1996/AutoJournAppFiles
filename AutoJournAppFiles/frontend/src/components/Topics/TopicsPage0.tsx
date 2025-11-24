import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Chart as ChartJS, ArcElement, Tooltip, Legend} from "chart.js";
import { Pie } from "react-chartjs-2";
import ChartDataLabels from "chartjs-plugin-datalabels";
import ReactMarkdown from "react-markdown";
import "./TopicsPage.css";

// Register required Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend, ChartDataLabels);

const TopicsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const { topics, originalText, topicsAndKeywords, keywords } = location.state || {};



  const [summaries, setSummaries] = useState<{ [key: string]: string }>({});
  const [selectedSummary, setSelectedSummary] = useState<string | null>(null);
  const [hoveredTopic, setHoveredTopic] = useState<string | null>(null); // Track the hovered topic
  const [loadingSummary, setLoadingSummary] = useState<boolean>(false); // Loading state for summary
  const [showTopicPopup, setShowTopicPopup] = useState<boolean>(false); // topic selection Popup visibility
  const [showStylePopup, setShowStylePopup] = useState<boolean>(false); // style selection popup
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false); // State for loading screen
  const [perspectives, setPerspectives] = useState<{ [key: string]: string }>({});
  const [loadingPerspectives, setLoadingPerspectives] = useState<boolean>(false);

  const [topicPerspectives, setTopicPerspectives] = useState<{ [topic: string]: any }>({});
  const [selectedTopicPerspectives, setSelectedTopicPerspectives] = useState<any | null>(null);

  
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
  ]; // List of styles

  // Fetch summaries when the component mounts
  useEffect(() => {
    const fetchSummaries = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8787/topics/Mistral/summaries");
        if (response.ok) {
          const data = await response.json();
          setSummaries(
            data.summaries.reduce(
              (acc: { [key: string]: string }, item: { topic: string; summary: string }) => {
                acc[item.topic] = item.summary;
                return acc;
              },
              {}
            )
          );
        } else {
          console.error("Failed to fetch summaries.");
        }
      } catch (error) {
        console.error("Error fetching summaries:", error);
      }
    };
    const fetchPerspectives = async () => {
      setLoadingPerspectives(true);
      try {
        const response = await fetch("http://127.0.0.1:8787/topics/Mistral/perspectives");
        if (response.ok) {
          const data = await response.json();
          const parsedPerspectives = Object.entries(data.perspectives).reduce(
            (acc: { [key: string]: string }, [key, value]: [string, any]) => {
              acc[`Perspective ${key}`] = `(${value.Stance}) ${value.Reason}`;
              return acc;
            },
            {}
          );
          setPerspectives(parsedPerspectives);
        } else {
          console.error("Failed to fetch perspectives.");
        }
      } catch (error) {
        console.error("Error fetching perspectives:", error);
      } finally {
        setLoadingPerspectives(false);
      }
    };

    fetchSummaries();
    fetchPerspectives();
  }, []);
  

  // Handle bubble click
  const handleBubbleClick = async (topicName: string) => {
    setLoadingSummary(true); // Set loading state
    try {
      const response = await fetch("http://127.0.0.1:8787/topics/Mistral/summaries");
      if (response.ok) {
        const data = await response.json();
        const updatedSummaries = data.summaries.reduce(
          (acc: { [key: string]: string }, item: { topic: string; summary: string }) => {
            acc[item.topic] = item.summary;
            return acc;
          },
          {}
        );
  
        setSummaries(updatedSummaries);
        setSelectedSummary(updatedSummaries[topicName] || "Summary not available.");
      } else {
        console.error("Failed to re-fetch summaries.");
        setSelectedSummary("Click on the Topic to see Summary");
      }
  
      // Fetch topic-specific perspectives directly from API and update both states
      const perspectiveResponse = await fetch("http://127.0.0.1:8787/topics/Mistral/perspectives");
      if (perspectiveResponse.ok) {
        const fullResponse = await perspectiveResponse.json();
        const perspectiveData = fullResponse.perspectives || fullResponse;
        //console.log("Topic clicked:", topicName);
        //console.log("All topic keys:", Object.keys(perspectiveData));
  
        setTopicPerspectives(perspectiveData); // cache it
        setSelectedTopicPerspectives(perspectiveData[topicName]); // directly use fetched data
      } else {
        console.error("Failed to fetch perspectives for topic.");
      }
    } catch (error) {
      console.error("Error fetching summaries or perspectives:", error);
      setSelectedSummary("An error occurred while fetching the summary.");
    } finally {
      setLoadingSummary(false); // Reset loading state
    }
  };
  

  // Prepare data for pie chart
  const chartData = {
    labels: Object.keys(topics),
    datasets: [
      {
        data: Object.values(topics),
        backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"], // Assign colors to topics
        hoverBackgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"],
      },
    ],
  };

  
  // Handle topic click
  const handleTopicClick = (topic: string) => {
    setSelectedTopic(topic);
    setShowTopicPopup(false);
    setShowStylePopup(true); // Open the style selection popup
  };
  // Handle style selection
  const handleStyleClick = async (style: string) => {
    //setShowStylePopup(false); // Close the style selection popup
    setIsLoading(true); // Show loading screen

    if (selectedTopic) {
      
      try {
        const summary = summaries[selectedTopic];
        const keywordsSent = keywords[selectedTopic];
        const requestBody = {
          topic: selectedTopic,
          summary,
          keywordsSent,
          style,
        };

        const response = await fetch("http://127.0.0.1:8787/generate-news", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });
        console.log("Data sent to backend", requestBody);
        console.log(response) 

        if (response.ok) {
          const { article } = await response.json();
          navigate("/news-article", { state: { article } });
        } else {
          console.error("Failed to generate news article.");
        }
      } catch (error) {
        console.error("Error generating news article:", error);
      }finally {
        setIsLoading(false); // Hide loading screen
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
      {/* First Column */}
      <div className="topics-left">
        <h1>Extracted Topics</h1>

        {/* Scrollable division for the original text */}
        <div className="original-text-container">
          <h2>Original Text:</h2>
          <div className="scrollable-text">
            <p>{originalText}</p>
          </div>
        </div>

        {/* New Topics List */}
        <div className="topics-list-container">
          
          <div className="division">
              <div className="subcolumn">
                 {/* Bubble visualization for topics */}
                 <h2>Suggested Topics</h2>
                 <p>Click on Each Topic to generate a Summary</p>
                 
                  <div className="topics-bubbles">
                    {Object.entries(topics).map(([topicName, percentage], index) => (
                      <div
                        key={index}
                        className="bubble"
                        
                        data-percentage={`${percentage}%`}
                        onClick={() => handleBubbleClick(topicName)} // Re-fetch summaries on click
                        onMouseEnter={() => setHoveredTopic(`${topicName} (${percentage}%)`)} // Set tooltip text on hover
                        onMouseLeave={() => setHoveredTopic(null)} // Clear tooltip text on mouse leave
                      >
                        <span className="bubble-text">{topicName}</span>
                      </div>
                    ))}
                    {hoveredTopic && <div className="tooltip">{hoveredTopic}</div>}
                  </div>
              </div>
              <div className="subcolumn">
                {/* Pie Chart */}
                <div className="chart-container">
                  <h2>Topic Distribution</h2>
                  <Pie data={chartData}/>
                </div>
              </div>
          </div>
          
        </div>

        <button className="back-button" onClick={() => navigate("/")}>
          Back to Home
        </button>
        <button className="generate-article-button" onClick={() => setShowTopicPopup(true)}>
          Generate News Article
        </button>
      </div>

      {/* Second Column */}
      <div className="topics-right">
        <h1>Summary</h1>
        <div className="summary-container">
         <div className="scrollable-text-summary">
          {loadingSummary ? (
            <p>Loading summary...</p>
          ) : selectedSummary ? (
            <ReactMarkdown>{selectedSummary}</ReactMarkdown>
          ) : (
            <div className="summary-placeholder">
              <p>Select a topic to view the summary here.</p>
            </div>
          )}
          </div>
        </div>
        <h2>Multiple Perspectives</h2>
          {selectedTopicPerspectives ? (
            <>
              <p><strong>Agree:</strong> {selectedTopicPerspectives.Agree}% &nbsp; | &nbsp;
                <strong>Disagree:</strong> {selectedTopicPerspectives.Disagree}%</p>
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

      {/*Topic Popup*/}
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

      {/* Style Selection Popup */}
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

    </div>
  );
};

export default TopicsPage;
