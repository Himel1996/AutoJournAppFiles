import React, { useState, useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import ReactModal from "react-modal"; 
import { useLocation, useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import "./NewsArticlePage.css";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import "jspdf-autotable";
import Chart from "chart.js/auto";
import html2canvas from "html2canvas";
import { marked } from "marked";
import { API_BASE_URL } from "../config";
//import ForceGraph2D from "react-force-graph-2d";

import * as d3 from "d3";
import { Network } from "vis-network/standalone";
export interface GraphViewHandle {
  exportAsImage: () => void;
  exportAsJSON: () => void;
}


//import convexHull from "convex-hull";

// Component to render community graph with vis-network
// Helper to generate convex hull path
/* const hullPath = (points: number[][]) => {
  if (points.length < 3) return null;
  const hullIndices = convexHull(points);
  const hullPoints = hullIndices.map(([i]: [number, number]) => points[i]);
  return d3.line().curve(d3.curveCardinalClosed.tension(0.75))(hullPoints as [number, number][]);
}; */

const GraphView1 = forwardRef(({ graphData }: { graphData: { nodes: any[]; links: any[] } }, ref) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const networkRef = useRef<Network | null>(null);

  useImperativeHandle(ref, () => ({
    exportAsImage: () => {
      if (!containerRef.current) return;
      html2canvas(containerRef.current).then((canvas) => {
        const link = document.createElement("a");
        link.download = "community-graph.png";
        link.href = canvas.toDataURL("image/png");
        link.click();
      });
    },
    exportAsJSON: () => {
      const nodesData = graphData.nodes.map((node) => ({
        id: node.id,
        community: node.community,
        label: node.label,
        sentiment: node.sentiment,
      }));
  
      const jsonStr = JSON.stringify(nodesData, null, 2);
      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
  
      const link = document.createElement("a");
      link.href = url;
      link.download = "community_nodes.json";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    },
  }));

  useEffect(() => {
    if (!containerRef.current) return;

    const seenCommunities = new Set<number>();
    const data = {
      nodes: graphData.nodes.map((node) => {
        const showLabel = !seenCommunities.has(node.community);
        if (showLabel) seenCommunities.add(node.community);

        return {
          id: node.id,
          label: showLabel ? node.label : "",
          group: node.community,
        };
      }),
      edges: graphData.links.map((link) => ({
        from: link.source,
        to: link.target,
        value: link.value,
        arrows: "",
        width: 1,
        color: { color: "#bbb" },
      })),
    };

    const options = {
      layout: {
        improvedLayout: true,
      },
      nodes: {
        shape: "dot",
        size: 10,
        font: {
          size: 52,
          color: "#333",
        },
      },
      edges: {
        smooth: true,
        color: "#ccc",
      },
      physics: {
        stabilization: true,
      },
    };

    networkRef.current = new Network(containerRef.current, data, options);
  }, [graphData]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "600px",
        border: "1px solid #ccc",
        borderRadius: "8px",
        backgroundColor: "#fff",
      }}
    />
  );
});

// 1) Type the prop as optional + nullable
type GraphPayload = { nodes: any[]; links: any[] };
const GraphView = forwardRef(
  ({ graphData }: { graphData?: GraphPayload | null }, ref) => {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const networkRef = useRef<Network | null>(null);

    useImperativeHandle(ref, () => ({
      exportAsImage: () => {
        if (!containerRef.current) return;
        html2canvas(containerRef.current).then((canvas) => {
          const link = document.createElement("a");
          link.download = "community-graph.png";
          link.href = canvas.toDataURL("image/png");
          link.click();
        });
      },
      exportAsJSON: () => {
        const nodes = Array.isArray(graphData?.nodes) ? graphData!.nodes : [];
        const nodesData = nodes.map((node) => ({
          id: node.id,
          community: node.community,
          label: node.label,
          sentiment: node.sentiment,
        }));
        const payload = {
          nodes: nodesData,
          links: (graphData as any)?.links ?? []
        };

        // const blob = new Blob([JSON.stringify(nodesData, null, 2)], {
        //   type: "application/json",
        // });
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "graph.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      },
    }));

    useEffect(() => {
      // Bail early until everything exists
      if (
        !containerRef.current ||
        !graphData ||
        !Array.isArray(graphData.nodes) ||
        !Array.isArray(graphData.links)
      ) {
        return;
      }

      const seenCommunities = new Set<number>();
      const data = {
        nodes: graphData.nodes.map((node) => {
          const showLabel = !seenCommunities.has(node.community);
          if (showLabel) seenCommunities.add(node.community);
          return {
            id: node.id,
            label: showLabel ? node.label : "",
            group: node.community,
            sentiment: node.sentiment,
          };
        }),
        edges: graphData.links.map((link) => ({
          from: link.source,
          to: link.target,
          value: link.value,
          arrows: "",
          width: 1,
          color: { color: "#bbb" },
        })),
      };

      const options = {
        layout: { improvedLayout: true },
        nodes: {
          shape: "dot",
          size: 10,
          font: { size: 52, color: "#333" },
        },
        edges: { smooth: true, color: "#ccc" },
        physics: { stabilization: true },
      };

      networkRef.current = new Network(containerRef.current, data, options);
    }, [graphData]);

    // Optional: render a simple placeholder while waiting for data
    if (!graphData || !Array.isArray(graphData.nodes)) {
      return (
        <div
          ref={containerRef}
          style={{
            width: "100%",
            height: "600px",
            border: "1px solid #ccc",
            borderRadius: "8px",
            backgroundColor: "#fff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#777",
          }}
        >
          Waiting for graph…
        </div>
      );
    }

    return (
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "600px",
          border: "1px solid #ccc",
          borderRadius: "8px",
          backgroundColor: "#fff",
        }}
      />
    );
  }
);





interface BiasTypeResult {
  sentence: string;
  bias_type: string;
  score: number;
}

// Define the graph data type
interface NodeType {
  id: string;
  community: number;
  label: string;
}

interface LinkType {
  source: string;
  target: string;
  value: number;
}

interface GraphDataType {
  nodes: NodeType[];
  links: LinkType[];
}

const NewsArticlePage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  //const { article: originalArticle, topic: topicName} = location.state || {};
  const {
    article: originalArticle,
    topic: topicName,
    originalText: originalText,
    mergedSummary: mergedSummary,
    perspectives: perspectives,
    allTopics: allTopics,
    topicProbabilities: topicProbabilities,
    allPerspectives: allPerspectives,
  } = location.state || {};
  const [articleText, setArticleText] = useState<string>(originalArticle || "");
  const [highlightedSentences, setHighlightedSentences] = useState<string[]>([]);
  const [biasScores, setBiasScores] = useState<number[]>([]);
  const [biasTypes, setBiasTypes] = useState<BiasTypeResult[]>([]);
  const [loadingBias, setLoadingBias] = useState(false);
  const [loadingTypes, setLoadingTypes] = useState(false);
  const [allSentences, setAllSentences] = useState<string[]>([]);
  const [selectedSentence, setSelectedSentence] = useState<string | null>(null);
  const [popupInfo, setPopupInfo] = useState<{
    label: string;
    typeScore: number;
    detectScore: number;
  } | null>(null);
  const [neutralizedSentences, setNeutralizedSentences] = useState<string[]>([]);
  const [showNeutralPopup, setShowNeutralPopup] = useState(false);
  const [biasMatchedSentences, setBiasMatchedSentences] = useState<string[]>([]);

  const [popupVisible, setPopupVisible] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const graphRef = useRef<GraphViewHandle>(null);


  interface BiasStat {
    sentence: string;
    bias_type: string;
    detect_score: number;
    type_score: number;
    neutral: string;
  }
  const combinedStats: BiasStat[] = highlightedSentences.map((sent, i) => ({
    sentence: sent,
    detect_score: biasScores[i],
    bias_type: biasTypes[i]?.bias_type || "Unknown",
    type_score: biasTypes[i]?.score || 0,
    neutral: neutralizedSentences[i] || "—",
  }));

  


  if (!originalArticle) {
    return (
      <div className="news-article-container">
        <h1>No Article Found</h1>
        <p>The article could not be loaded. Please try again.</p>
        <button onClick={() => navigate("/")}>Go Back to Home</button>
      </div>
    );
  }
    const handleOpenEchoChamber = () => {
      navigate("/echo-chamber", {
        state: {
          article: articleText, // the generated news article
          graphData, // output from your detectEchoChamber() function
          originalText,
          topicName,
          topicProbabilities,
          allPerspectives,
          allTopics,
          summary: mergedSummary,
          biasStats: combinedStats
        }
      });
    };
  //helper functions
  const fixMarkdownHeadings = (text: string): string => {
    return text
      // Ensure two newlines before the heading
      .replace(/([^\n])(\n)?(#+\s)/g, (_, prev, newline, heading) => {
        return `${prev}\n\n${heading}`;
      })
      // Ensure two newlines after heading before first capitalized word (e.g., "## Heading\nThis..." → "## Heading\n\nThis...")
      .replace(/(#+\s[^\n]+)(\n)([A-Z])/g, (_, headingLine, newline, firstLetter) => {
        return `${headingLine}\n\n${firstLetter}`;
      });
  };
  const cleanSentence = (sentence: string): string => {
    return sentence
      .replace(/\*\*/g, "")           // Remove bold markers
      .replace(/\n+/g, " ")           // Replace newlines with space
      .replace(/\s+/g, " ")           // Normalize multiple spaces
      .trim();
  };
  const stripHeadings = (sentence: string): string => {
    return sentence
      .replace(/^\s*\*\*.*\*\*\s*$/gm, "") // Remove lines that are just **bold**
      .replace(/\*\*/g, "")               // Remove bold within sentence if any left
      .replace(/\n+/g, " ")               // Replace newlines with space
      .replace(/\s+/g, " ")               // Normalize spaces
      .trim();
  };
  
  const renderMarkdownToPDF = (doc: jsPDF, markdown: string, startY = 30, maxWidth = 180) => {
    const tokens = marked.lexer(markdown);
    let y = startY;
  
    tokens.forEach((token) => {
      switch (token.type) {
        case "heading":
          doc.setFontSize(token.depth === 1 ? 16 : token.depth === 2 ? 14 : 12);
          doc.setFont("helvetica", "bold");
          doc.text(token.text, 14, y);
          y += 8;
          break;
  
        case "paragraph":
          doc.setFontSize(10);
          doc.setFont("helvetica", "normal");
  
          // Split paragraph text into lines
          const lines = doc.splitTextToSize(token.text, maxWidth);
          doc.text(lines, 14, y);
          y += lines.length * 6;
          break;
  
        case "list":
          token.items.forEach((item: any) => {
            const lines = doc.splitTextToSize(`• ${item.text}`, maxWidth);
            doc.text(lines, 14, y);
            y += lines.length * 6;
          });
          break;
  
        case "code":
          doc.setFont("courier", "normal");
          doc.setFontSize(9);
          const codeLines = doc.splitTextToSize(token.text, maxWidth);
          doc.text(codeLines, 14, y);
          y += codeLines.length * 6;
          break;
  
        // Add more token types as needed
      }
  
      // Add new page if content overflows
      if (y > 270) {
        doc.addPage();
        y = 30;
      }
    });
  };

  //detect echo chambers
  const detectEchoChambers = async () => {
    const response = await fetch("${API_BASE_URL}/echo-chamber", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ originalText, allTopics }),
    });
    const data = await response.json();
    console.log("label: ", data)
    setGraphData(data);
    setPopupVisible(true);
  };

  //neutralize
  const neutralizeBiasedSentences = async () => {
    try {
      //const cleaned = highlightedSentences.map(stripHeadings);
      
      //console.log("Cleaned for neutralizer:", cleaned);
      const response = await fetch("${API_BASE_URL}/neutralize-bias", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentences: highlightedSentences }),
      });
      const data = await response.json();
      setNeutralizedSentences(data.neutralized_sentences);
    } catch (err) {
      console.error("Error neutralizing bias:", err);
    }
  };
  //bias detect
  const detectBias = async (): Promise<string[]> => {
    setLoadingBias(true);
    try {
      const response = await fetch("${API_BASE_URL}/detect-bias", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ article: articleText }),
      });
      const data = await response.json();
      const allSentencesFromBackend: string[] = data.all_sentences;
      //directly from backend
      const biasedSentences: string[] = data.biased_sentences;
      console.log("bias: ", biasedSentences)
      const scores: number[] = data.scores;
      setHighlightedSentences(biasedSentences);
      //console.log(biasedSentences)
      setBiasScores(scores);

      //filtering in frontend
      const flags: { sentence: string; biased: boolean; score: number }[] = data.bias_flags;
      const highlighted = flags.filter(f => f.biased).map(f => f.sentence);
      //setHighlightedSentences(highlighted);
      //setBiasScores(flags.filter(f => f.biased).map(f => f.score));

      setAllSentences(allSentencesFromBackend);

      setBiasMatchedSentences(biasedSentences); // the actual clean tokenized ones used for bias detection
      setAllSentences(allSentencesFromBackend); // for rendering
     
      return biasedSentences;  //highlighted
    } catch (err) {
      console.error("Bias detection failed:", err);
      return [];
    } finally {
      setLoadingBias(false);
    }
  };
 // bias classify
  const detectBiasTypes = async (sentences: string[]) => {
    setLoadingTypes(true);
    try {
      const response = await fetch("${API_BASE_URL}/bias-type", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentences }),
      });
      const data = await response.json();
      setBiasTypes(data.bias_types);
    } catch (err) {
      console.error("Bias type detection failed:", err);
    } finally {
      setLoadingTypes(false);
    }
  };
 //bias highlight
  const renderWithBiasHighlights = (article: string) => {
    /* if (!Array.isArray(highlightedSentences) || highlightedSentences.length === 0) {
      return <ReactMarkdown>{article}</ReactMarkdown>;
    } */
    if (!Array.isArray(allSentences) || allSentences.length === 0) {
      return <ReactMarkdown>{article}</ReactMarkdown>;
    }

    const normalize = (str: string) =>
      str.replace(/\*\*/g, "")
        .replace(/\n+/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .toLowerCase();

    const isHeading = (s: string) => /^\s*\*\*.*\*\*\s*$/.test(s.trim());

    //const sentences = article.match(/[\s\S]*?[.!?]+(?=\s|$)/g) || [article];
    const sentences = allSentences;
    console.log("all sents:", allSentences);

    return sentences.map((sentence: string, index: number) => {
      const trimmed = sentence.trim();
      
      //const biasIndex = highlightedSentences.findIndex(s => s.trim() === trimmed);
      const normalizedSentence = normalize(trimmed);
      //console.log(sentence)
      //console.log(normalizedSentence)
      //console.log(normalize(highlightedSentences[0]).trim())
      
      //const biasIndex = highlightedSentences.findIndex(s => normalize(s) === normalizedSentence);
      const biasIndex = biasMatchedSentences.findIndex(s => normalize(s) === normalizedSentence);

      const isBiased = biasIndex !== -1;

      const handleClick = () => {
        if (isBiased && trimmed) {
          const biasTypeInfo = biasTypes?.[biasIndex];
          if (biasTypeInfo) {
            setSelectedSentence(trimmed);
            setPopupInfo({
              label: biasTypeInfo.bias_type ?? "Unknown",
              typeScore: biasTypeInfo.score ?? 0,
              detectScore: biasScores?.[biasIndex] ?? 0,
            });
          }
        }
      };

      return (
        <span
          key={index}
          className={isBiased ? "biased-sentence clickable" : ""}
          onClick={handleClick}
          style={{ cursor: isBiased ? "pointer" : "default" }}
        >
          <ReactMarkdown components={{ p: 'span' }}>{sentence}</ReactMarkdown>{" "}
        </span>
      );
    });
  };
 // new article neutralize
//  const renderNeutralizedArticle = () => {
//   const normalize = (str: string) =>
//     str.replace(/\*\*/g, "").replace(/\n+/g, " ").replace(/\s+/g, " ").trim().toLowerCase();

//   const isMarkdownHeader = (text: string) =>
//     /^\s*(\*{2}.+\*{2}|#{1,6}\s+.+)\s*$/gm.test(text.trim());

//   let updatedSentences = allSentences.map((sent) => {
//     const norm = normalize(sent);
//     const matchIdx = highlightedSentences.findIndex((b) => normalize(b) === norm);
//     const replacement = matchIdx !== -1 ? neutralizedSentences[matchIdx] : sent;

//     // If the replacement is a markdown header, append two newlines
//     if (isMarkdownHeader(replacement)) {
//       return `${replacement.trim().toUpperCase()}\n\n`;
//     }
//     //console.log(allPerspectives);
//     return replacement;
//   });

//   return updatedSentences.join(" ");
// };

const renderNeutralizedArticle = () => {
  const normalize = (str: string) =>
    str?.replace(/\*\*/g, "").replace(/\n+/g, " ").replace(/\s+/g, " ").trim().toLowerCase();

  const isMarkdownHeader = (text: string) =>
    typeof text === "string" && /^\s*(\*{2}.+\*{2}|#{1,6}\s+.+)\s*$/gm.test(text.trim());

  let updatedSentences = allSentences.map((sent) => {
    const norm = normalize(sent);
    const matchIdx = highlightedSentences.findIndex((b) => normalize(b || "") === norm);
    const replacement = matchIdx !== -1 ? neutralizedSentences[matchIdx] || sent : sent;

    if (isMarkdownHeader(replacement)) {
      return `${replacement.trim().toUpperCase()}\n\n`;
    }

    return replacement;
  });

  return updatedSentences.join(" ");
};


//save neutral version
const neutralizedArticle = renderNeutralizedArticle();


  //bias stats working pdf function:
  const handleDownloadPDF = async () => {
    await exportBiasStatisticsPDF(combinedStats, topicName, allSentences.length);
  };

  const exportBiasStatisticsPDF = async (stats: BiasStat[], topicTitle: string, totalSentences: number) => {
    const doc = new jsPDF();
  
    const date = new Date().toLocaleDateString();
    const timestamp = new Date().toISOString().slice(0, 10);

    const biasedCount = stats.length;
    const biasPercent = ((biasedCount / totalSentences) * 100).toFixed(2);
  
    doc.setFontSize(16);
    doc.text("Bias Analysis Report", 14, 20);
    doc.setFontSize(12);
    doc.text(`Topic: ${topicTitle}`, 14, 30);
    doc.text(`Date: ${date}`, 14, 38);
  
    const headers = [
      "Biased Sentence",
      "Bias Confidence",
      "Bias Type",
      "Type Score",
      "Neutralized Sentence",
    ];
  
    const rows = stats.map((stat) => [
      stat.sentence,
      stat.detect_score.toFixed(2),
      stat.bias_type,
      stat.type_score.toFixed(2),
      stat.neutral,
    ]);
  
    autoTable(doc,{
      head: [headers],
      body: rows,
      startY: 45,
      styles: { fontSize: 9, cellPadding: 2 },
      columnStyles: {
        0: { cellWidth: 60 }, // Biased Sentence
        4: { cellWidth: 60 }, // Neutralized Sentence
      },
      theme: "striped",
    });

    // Add summary at bottom
    const finalY = (doc as any).lastAutoTable.finalY || 60;
    doc.text(`Total Sentences: ${totalSentences}`, 14, finalY + 10);
    doc.text(`Biased Sentences: ${biasedCount}`, 14, finalY + 18);
    doc.text(`Bias Percentage: ${biasPercent}%`, 14, finalY + 26);


    // Add pie chart
    //const canvas = document.createElement("canvas");
    let canvas = document.getElementById("bias-pie-chart") as HTMLCanvasElement;

    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.id = "bias-pie-chart";
      document.body.appendChild(canvas);
    }

    canvas.width = 600;
    canvas.height = 600;

    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    new Chart(canvas, {
      type: "pie",
      data: {
        labels: ["Biased", "Unbiased"],
        datasets: [{
          data: [biasedCount, totalSentences - biasedCount],
          backgroundColor: ["#ff6384", "#36a2eb"],
          borderColor: ["#ff6384", "#36a2eb"],  // match borderColor to backgroundColor
          borderWidth: 2, // slightly increase to force overlap
        }],
      },
      options: {
        responsive: false,
        animation: false,  // ensures static rendering for screenshot
        plugins: {
          legend: { position: "bottom" },
          title: {
            display: true,
            text: "Bias Distribution",
            font: { size: 18 }
          }
        },
        layout: {
          padding: 10
        }
      }
    });

    // Add page for chart
    doc.addPage();
    doc.setFontSize(14);
    doc.text("Bias Distribution Chart", 105, 20, { align: "center" });

    await new Promise((resolve) => setTimeout(resolve, 500)); // Let chart render

    // Capture and add the chart
    
    const chartImage = await html2canvas(canvas).then(canvas => canvas.toDataURL("image/png"));

    // Insert image centered
    const imgProps = doc.getImageProperties(chartImage);
    const pdfWidth = doc.internal.pageSize.getWidth();
    const chartWidth = 140;
    const chartHeight = (imgProps.height * chartWidth) / imgProps.width;
    const xPos = (pdfWidth - chartWidth) / 2;


    doc.addImage(chartImage, "PNG", xPos, 30, chartWidth, chartHeight);

    document.body.removeChild(canvas); // Clean up
  
    doc.save(`bias_statistics_${topicTitle}_${timestamp}.pdf`);
  };

  //function for full report download:
  //bias stats working pdf function:
  const handleFullReportPDF = async () => {
    try {
      await generateFullReportPDF({
        originalText,
        originalArticle,
        topicName,
        mergedSummary,
        perspectives,
        allTopics,
        topicProbabilities,
        allPerspectives,
        biasStats: combinedStats, // or whatever your variable is
        totalSentences: allSentences.length, // also ensure allSentences is available
        neutralizedArticle,
      });
    } catch (error) {
      console.error("Error generating full report PDF:", error);
    }
  };
  
  type FullReportProps = {
    originalText: string;
    originalArticle: string;
    topicName: string;
    mergedSummary: string;
    perspectives: Record<string, any>;
    allTopics: Record<string, number>;
    topicProbabilities: number[];
    allPerspectives: Record<string, any>;
    biasStats: BiasStat[];
    totalSentences: number;
    neutralizedArticle: string;
  };
  
  const generateFullReportPDF = async ({
    originalText,
    originalArticle,
    topicName,
    mergedSummary,
    perspectives,
    allTopics,
    topicProbabilities,
    allPerspectives,
    biasStats,
    totalSentences,
    neutralizedArticle
  }: FullReportProps) => {
    const doc = new jsPDF();
    const timestamp = new Date().toISOString().slice(0, 10);

    const biasedCount = biasStats.length;
    const biasPercent = ((biasedCount / totalSentences) * 100).toFixed(2);

    //var pageHeight = doc.internal.pageSize.height || doc.internal.pageSize.getHeight();
    //var pageWidth = doc.internal.pageSize.width || doc.internal.pageSize.getWidth();
  
    // Title Page
    doc.setTextColor(25, 50, 100); // Dark blue
    doc.setFont("times", "italic");
    doc.setFontSize(28);
    doc.text("Automated Journalist Report", doc.internal.pageSize.getWidth() / 2, 100, {align: 'center'});
    doc.setFontSize(18);
    doc.text(`Generated on: ${timestamp}`, doc.internal.pageSize.getWidth() / 2, 110, {align: 'center'});

    doc.setDrawColor(25, 50, 100);
    doc.line(50, 115, 160, 115);

    // Reset to default styles after title page
    doc.addPage();
    doc.setFont("helvetica", "normal"); // Default font
    doc.setTextColor(0, 0, 0);          // Default black text


    //Original Text
    doc.setFontSize(14);
    doc.text("Original Extracted Text", 14, 20);
    doc.setFontSize(10);
    doc.text(originalText, 14, 30, { maxWidth: 180 });
    //renderMarkdownToPDF(doc, originalText, 30, 180);
    doc.addPage();
  
    // Topics Overview
    doc.setFontSize(14);
    doc.text("Generated Topics", 14, 20);
    const topicEntries = Object.entries(allTopics);
    topicEntries.forEach(([topic, score], i) => {
      doc.text(`- ${topic} (${score.toFixed(2)}%)`, 14, 30 + i * 8);
    });
  
    // Topic Pie Chart
    const topicCanvas = document.createElement("canvas");
    topicCanvas.width = 400;
    topicCanvas.height = 400;
    document.body.appendChild(topicCanvas);
  
    new Chart(topicCanvas, {
      type: "pie",
      data: {
        labels: topicEntries.map(([t]) => t),
        datasets: [{
          data: topicEntries.map(([_, p]) => p),
          backgroundColor: ["#ff6384", "#36a2eb", "#ffcd56", "#4bc0c0", "#9966ff", "#ff9f40"],
          borderColor: ["#ff6384", "#36a2eb", "#ffcd56", "#4bc0c0", "#9966ff", "#ff9f40"],
          borderWidth: 2,
        }]
      },
      options: {
        responsive: false,
        animation: false,
        plugins: {
          title: { display: true, text: "Topic Distribution" },
          legend: { position: "bottom" }
        }
      }
    });
  
    await new Promise(res => setTimeout(res, 400));
    const topicImg = await html2canvas(topicCanvas).then(c => c.toDataURL("image/png"));
    doc.addPage();
    doc.addImage(topicImg, "PNG", 30, 30, 150, 150);
    document.body.removeChild(topicCanvas);
  
    // Perspectives Table
    doc.addPage();
    doc.setFontSize(14);
    doc.text("Topic Perspectives", 14, 15);

    for (const [topic, details] of Object.entries(allPerspectives)) {
      doc.setFontSize(12);
      const finalY = (doc as any).lastAutoTable?.finalY || 30;
      doc.text(`Topic: ${topic}`, 14, finalY + 10);

      const perspectiveRows = Object.entries(details.Perspectives).map(([_, val]: [string, any]) => [
        val.Stance,
        val.Reason,
        val.Criteria.join(", ")
      ]);
      //doc.text(`Agree: ${details.Agree}% | Disagree: ${details.Disagree}%`, 14, finalY + 6);

      autoTable(doc, {
        head: [["Stance", "Reason", "Criteria"]],
        body: perspectiveRows,
        startY: (doc as any).lastAutoTable?.finalY + 20 || 45,
        styles: { fontSize: 9 },
        columnStyles: {
          0: { cellWidth: 25 },
          1: { cellWidth: 85 },
          2: { cellWidth: 75 }
        },
        margin: { top: 10 },
      });

      // Add a new page if content is getting too long
      /* if ((doc as any).lastAutoTable.finalY > 260) {
        doc.addPage();
      } */
    }
  
    // Summary
    doc.addPage();
    doc.setFontSize(14);
    doc.text(`Summary of Selected Topic: ${topicName}`, 14, 20);
    doc.setFontSize(12);
    doc.text(mergedSummary, 14, 30, { maxWidth: 180 });

    //Generated News Article
    doc.addPage();
    doc.setFontSize(14);
    doc.text("Original Generated News Article", 14, 20);
    doc.setFontSize(12);
    //doc.text(originalArticle, 14, 30, { maxWidth: 180 });
    renderMarkdownToPDF(doc, originalArticle, 30, 180);

    
  
    // Bias Stats Table
    doc.addPage();
    doc.setFontSize(14);
    doc.text("Bias Statistics", 14, 20);
    const biasTable = biasStats.map(stat => [
      stat.sentence,
      stat.detect_score.toFixed(2),
      stat.bias_type,
      stat.type_score.toFixed(2),
      stat.neutral
    ]);
  
    autoTable(doc, {
      head: [["Biased Sentence", "Confidence", "Type", "Type Score", "Neutralized"]],
      body: biasTable,
      startY: 30,
      styles: { fontSize: 9, cellPadding: 2 }
    });
    // Add bias summary at bottom
    const finalY = (doc as any).lastAutoTable.finalY || 60;
    doc.text(`Total Sentences: ${totalSentences}`, 14, finalY + 10);
    doc.text(`Biased Sentences: ${biasedCount}`, 14, finalY + 18);
    doc.text(`Bias Percentage: ${biasPercent}%`, 14, finalY + 26);
  
    // Bias Chart
    const biasCanvas = document.createElement("canvas");
    biasCanvas.width = 400;
    biasCanvas.height = 400;
    document.body.appendChild(biasCanvas);
  
    new Chart(biasCanvas, {
      type: "pie",
      data: {
        labels: ["Biased", "Unbiased"],
        datasets: [{
          data: [biasStats.length, totalSentences - biasStats.length],
          backgroundColor: ["#f44336", "#4caf50"],
          borderColor: ["#f44336", "#4caf50"],
          borderWidth: 2,
        }]
      },
      options: {
        responsive: false,
        animation: false,
        plugins: {
          title: { display: true, text: "Bias Distribution" },
          legend: { position: "bottom" }
        }
      }
    });
  
    await new Promise(res => setTimeout(res, 400));
    const biasImg = await html2canvas(biasCanvas).then(c => c.toDataURL("image/png"));
    doc.addPage();
    doc.addImage(biasImg, "PNG", 30, 30, 150, 150);
    document.body.removeChild(biasCanvas);

    // Final Bias-Neutralized Article
    doc.addPage();
    doc.setFontSize(14);
    doc.text("Bias-Neutralized Article", 14, 20);
    doc.setFontSize(12);

    // Render markdown version properly
    renderMarkdownToPDF(doc, neutralizedArticle, 30, 180);
  
    // Save the PDF
    doc.save(`full_report_${topicName}_${timestamp}.pdf`);
  };
  

  return (
    <div className="news-article-container">
      {/* <button
        style={{
          position: "absolute",
          right: 20,
          top: -20,
          padding: "8px 12px",
          background: "#333",
          color: "#fff",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer"
        }}
        onClick={detectEchoChambers}
      >
        Detect Echo Chamber
      </button> */}
      <div className="news-article-left">
        <div className="scrollable-text-summary">
          <h1>Generated News Article</h1>
          <div className="news-article-content">{renderWithBiasHighlights(articleText)}</div>
        </div>

        <div className="button-group">
          <button
            className="btn back-button2"
            onClick={async () => {
              const sentences = await detectBias();
              if (sentences.length > 0) {
                await detectBiasTypes(sentences);
              }
            }}
            disabled={loadingBias || loadingTypes}
          >
            {loadingBias || loadingTypes ? "Detecting..." : "Detect Bias"}
          </button>

          <button
            className="btn download-button"
            onClick={neutralizeBiasedSentences}
            disabled={highlightedSentences.length === 0}
          >
            Neutralize Bias
          </button>

          <button
            className="btn download-button"
            onClick={() => setShowNeutralPopup(true)}
            disabled={neutralizedSentences.length === 0}
          >
            Replace Bias
          </button>


          {/* <button
            className="btn download-button"
            //onClick={generatePDF}
            onClick={handleDownloadPDF}
            disabled={highlightedSentences.length === 0 || biasTypes.length === 0}
          >
            Download Bias Report
          </button> */}

          <button
            className="btn back-button1"
            //onClick={generatePDF}
            onClick={handleFullReportPDF}
            //disabled={highlightedSentences.length === 0 || biasTypes.length === 0}
          >
            Full Report
          </button>

    
          {/* <button className="btn back-button1" onClick={() => navigate("/")}>Back to Home</button> */}
        </div>

        {showNeutralPopup && (
          <div className="bias-popup">
            <h2>Neutralized Article</h2>
            <div className="scrollable-text-summary">
              <div className="news-article-content">
              <ReactMarkdown>{fixMarkdownHeadings(renderNeutralizedArticle())}</ReactMarkdown>
              </div>
            </div>
            <button className="btn back-button" onClick={() => setShowNeutralPopup(false)}>Close</button>
          </div>
        )}

        
        {/* Echo Chamber Popup */}
        {popupVisible && (
          <div
            style={{
              position: "fixed",
              top: "5%",
              left: "5%",
              width: "90%",
              height: "90%",
              backgroundColor: "#fff",
              border: "2px solid #ccc",
              borderRadius: "10px",
              padding: "20px",
              zIndex: 9999,
              boxShadow: "0 0 20px rgba(0,0,0,0.3)",
              overflow: "auto",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
              <h2>Echo Chamber Graph</h2>
              <button
                onClick={() => graphRef.current?.exportAsImage()}
                style={{
                  padding: "5px 10px",
                  fontSize: "16px",
                  background: "#2e7d32",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  marginLeft: "10px"
                }}
              >
                Download Image
              </button>

              <button
                onClick={() => graphRef.current?.exportAsJSON()}
                style={{
                  padding: "5px 10px",
                  fontSize: "16px",
                  background: "#1976d2",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Download JSON
              </button>


              <button
                onClick={() => setPopupVisible(false)}
                style={{
                  padding: "5px 10px",
                  fontSize: "16px",
                  background: "crimson",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Close
              </button>
            </div>

            {graphData ? (
              <>
                <div style={{ display: "flex", height: "100%" }}>
                    {/* Graph on the left */}
                    <div style={{ flex: 3 }}>
                      <GraphView ref={graphRef} graphData={graphData as GraphDataType} />
                    </div>

                    {/* Legend on the right */}
                    <div style={{ flex: 1, paddingLeft: "20px" }}>
                      <strong>Detected Communities:</strong>
                      <ul style={{ paddingLeft: "0", listStyleType: "none" }}>
                        {Array.from(
                          new Map(((graphData as GraphDataType)?.nodes ?? []).map((n) => [n.community, n.label])).entries()
                        ).map(([communityId, label]) => (
                          <li key={String(communityId)} style={{ marginBottom: "8px", display: "flex", alignItems: "center" }}>
                            <span
                              style={{
                                display: "inline-block",
                                width: 14,
                                height: 14,
                                backgroundColor: d3.schemeCategory10[Number(communityId) % 10],
                                borderRadius: "50%",
                                marginRight: 8,
                              }}
                            />
                            {label}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

              </>
            ) : (
              <p>Loading graph...</p>
            )}
          </div>
        )}





        {popupInfo && selectedSentence && (
          <div className="bias-popup">
            <p><strong>Sentence:</strong><br /> {selectedSentence}</p>
            <p><strong>Bias Detection Score:</strong> {popupInfo.detectScore.toFixed(2)}</p>
            <p><strong>Bias Type:</strong> {popupInfo.label}</p>
            <p><strong>Type Confidence:</strong> {popupInfo.typeScore.toFixed(2)}</p>
            <button className="btn back-button" onClick={() => setPopupInfo(null)}>Close</button>
          </div>
        )}
      </div>
      <div className="news-article-right">
        <h2>Neutralized Sentences</h2>
        {neutralizedSentences.length === 0 ? (
          <p className="neutral-info">No neutralized content yet. Click 'Neutralize Bias' to process.</p>
        ) : (
          <ul className="neutral-list">
            {neutralizedSentences.map((s, idx) => (
              <li key={idx}>{s}</li>
            ))}
          </ul>
        )}
        {/* <button
          className="echo-chamber-button"
          onClick={handleOpenEchoChamber}
          title="Open Echo Chamber Analysis"
        >
          ➤ EchoChamber
        </button> */}
      </div>
      
    </div>
  );
};

export default NewsArticlePage;
