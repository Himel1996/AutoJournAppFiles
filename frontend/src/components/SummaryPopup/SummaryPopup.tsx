import React, { useEffect, useState } from "react";
import "./SummaryPopup.css";

interface SummaryPopupProps {
  topic: string;
  onClose: () => void;
  onSaveMergedSummary: (topic: string, summary: string) => void;
}

const SummaryPopup: React.FC<SummaryPopupProps> = ({ topic, onClose, onSaveMergedSummary }) => {
  const [agreeSummary, setAgreeSummary] = useState<string>("");
  const [disagreeSummary, setDisagreeSummary] = useState<string>("");
  const [mergedSummary, setMergedSummary] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchSummaries = async () => {
      try {
        const response = await fetch("http://72.62.44.22:8000/summarize-perspectives", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic }),
        });

        if (response.ok) {
          const data = await response.json();
          setAgreeSummary(data.agree);
          setDisagreeSummary(data.disagree);
          console.log("Agree Summary:", data.agree);

        } else {
          console.error("Failed to fetch summaries.");
        }
      } catch (error) {
        console.error("Error fetching summaries:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchSummaries();
  }, [topic]);

  const handleMerge = async () => {
    try {
      const response = await fetch("http://72.62.44.22:8000/merge-summaries", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, agree: agreeSummary, disagree: disagreeSummary }),
      });
      if (response.ok) {
        const data = await response.json();
        setMergedSummary(data.merged);
        onSaveMergedSummary(topic, data.merged);
      }
    } catch (error) {
      console.error("Error merging summaries:", error);
    }
  };

  return (
    <div className="popup-overlay">
      <div className="popup-content">
        <h2>Summaries for Topic: {topic}</h2>
        {loading ? (
          <p>Loading summaries...</p>
        ) : (
          <>
            <div className="summary-block">
              <h3>Agree Summary</h3>
              <p>{agreeSummary}</p>
            </div>
            <div className="summary-block">
              <h3>Disagree Summary</h3>
              <p>{disagreeSummary}</p>
            </div>
            <button onClick={handleMerge}>Merge Summaries</button>
            {mergedSummary && (
              <div className="summary-block">
                <h3>Merged Summary</h3>
                <p>{mergedSummary}</p>
              </div>
            )}
          </>
        )}
        <button onClick={onClose} className="back-button">Close</button>
      </div>
    </div>
  );
};

export default SummaryPopup;
