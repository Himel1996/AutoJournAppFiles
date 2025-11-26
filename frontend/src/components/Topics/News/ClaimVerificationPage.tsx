// ClaimVerificationPage.tsx
import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import axios from "axios";
import "./ClaimVerificationPage.css";
import { API_BASE_URL } from "../config";
const ClaimVerificationPage = () => {
  const location = useLocation();
  const [article, setArticle] = useState<string>("");

    useEffect(() => {
    try {
        const parsed = JSON.parse(window.name || "{}");
        if (parsed.article) {
        setArticle(parsed.article);
        }
    } catch (err) {
        console.error("Error parsing article from window.name", err);
    }
    }, []);

  const [claims, setClaims] = useState<string[]>([]);
  const [selectedClaim, setSelectedClaim] = useState<string | null>(null);
  const [verificationResult, setVerificationResult] = useState<any | null>(null);
  const [showArticle, setShowArticle] = useState(false);
  const [loading, setLoading] = useState(false);

  const [proposedEdit, setProposedEdit] = useState<{
    target_sentence: string;
    corrected_sentence: string;
  } | null>(null);
  const [loadingProposal, setLoadingProposal] = useState(false);
  const [loadingApply, setLoadingApply] = useState(false);


  useEffect(() => {
    const fetchClaims = async () => {
      try {
        const res = await axios.post("${API_BASE_URL}/extract-claims", {
          article: article,
        });
        setClaims(res.data.claims);
      } catch (err) {
        console.error("Error fetching claims:", err);
      }
    };
  
    if (article) fetchClaims();
  }, [article]);

  const handleVerify = async () => {
    if (!selectedClaim) return;
    try {
      setLoading(true);
      const res = await axios.post("${API_BASE_URL}/verify-claim", {
        claim: selectedClaim,
      });
      setVerificationResult(res.data);
      setProposedEdit(null); // reset previous proposal
    } catch (err) {
      console.error("Error verifying claim:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestCorrection = async () => {
    if (!selectedClaim || !verificationResult) return;
    try {
      setLoadingProposal(true);
      const res = await axios.post("${API_BASE_URL}/propose-correction", {
        claim: selectedClaim,
        label: verificationResult.label,
        summary: verificationResult.summary,
        article
      });

      if (res.data.can_edit) {
        setProposedEdit({
          target_sentence: res.data.target_sentence,
          corrected_sentence: res.data.corrected_sentence
        });
      } else {
        setProposedEdit(null);
        alert(res.data.reason || "No edit proposed.");
      }
    } catch (err) {
      console.error("Error proposing correction:", err);
    } finally {
      setLoadingProposal(false);
    }
  };
  const handleApplyEdit = async () => {
    if (!proposedEdit) return;
    try {
      setLoadingApply(true);
      const res = await axios.post("${API_BASE_URL}/apply-edit", {
        article,
        target_sentence: proposedEdit.target_sentence,
        corrected_sentence: proposedEdit.corrected_sentence
      });
      if (res.data.updated_article) {
        setArticle(res.data.updated_article);
        setProposedEdit(null);
        alert("Edit applied to article.");
      }
    } catch (err) {
      console.error("Error applying edit:", err);
    } finally {
      setLoadingApply(false);
    }
  };
  const downloadArticle = () => {
    const blob = new Blob([article], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "news_article.md";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="claim-verification-container">
      <div className="left-panel">
        <h3>🧾 Extracted Claims</h3>
        {claims.length === 0 && <p>No claims found.</p>}
        <ul>
          {claims.map((claim, idx) => (
            <li
              key={idx}
              // onClick={() => setSelectedClaim(claim)}
              onClick={() => {
                setSelectedClaim(claim);
                setVerificationResult(null);
                setProposedEdit(null);
              }}
              className={selectedClaim === claim ? "selected" : ""}
            >
              {claim}
            </li>
          ))}
        </ul>
        <button
          onClick={handleVerify}
          disabled={!selectedClaim || loading}
          className="verify-btn"
        >
          {loading ? "Verifying..." : "🔍 Verify Claim"}
        </button>
        <button onClick={() => setShowArticle(!showArticle)} className="article-btn">
          📄 {showArticle ? "Hide" : "Show"} Original/Corrected Article
        </button>
        {showArticle && <div className="article-box">{article}</div>}

        <button onClick={downloadArticle} className="download-btn">
          ⬇️ Download Corrected Article
        </button>
      </div>

      <div className="right-panel">
        <h3>🔎 Verification Result</h3>
        {verificationResult ? (
          <div>
            <p>
              <strong>Claim:</strong> {selectedClaim}
            </p>
            <p>
              <strong>Label:</strong>{" "}
              <span className={`label label-${verificationResult.label}`}>
                {verificationResult.label.toUpperCase()}
              </span>
            </p>
            <p>
              <strong>Evidence Summary:</strong>
              <br />
              {verificationResult.summary}
            </p>

            {/* Evidence Sources list */}
            {verificationResult.evidence_sources?.length > 0 && (
              <div className="evidence-sources">
                <strong>Sources:</strong>
                <ul>
                  {verificationResult.evidence_sources.map((s: any, i: number) => (
                    <li key={i}>
                      <a href={s.link} target="_blank" rel="noreferrer">
                        {s.source || s.title || s.link}
                      </a>
                      {s.publishedAt && (
                        <> — <em>{new Date(s.publishedAt).toLocaleDateString()}</em></>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {(verificationResult.label === "false" ||
              verificationResult.label === "pants-on-fire" ||
              verificationResult.label === "barely true") && (
              <button
                onClick={handleSuggestCorrection}
                disabled={loadingProposal}
                className="suggest-btn"
              >
                {loadingProposal ? "Proposing..." : "✏️ Suggest Correction"}
              </button>
            )}

            {proposedEdit && (
              <div className="proposal-box">
                <p><strong>Original sentence:</strong><br />{proposedEdit.target_sentence}</p>
                <p><strong>Proposed correction:</strong><br />{proposedEdit.corrected_sentence}</p>
                <button
                  onClick={handleApplyEdit}
                  disabled={loadingApply}
                  className="apply-btn"
                >
                  {loadingApply ? "Applying..." : "✅ Apply Edit"}
                </button>
              </div>
            )}


          </div>
        ) : (
          <p>No verification result yet.</p>
        )}
      </div>
    </div>
  );
};

export default ClaimVerificationPage;
