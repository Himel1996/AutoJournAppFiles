import { Dialog, DialogTitle, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from "@mui/material";
import { Samsum } from "../../backend-objects/Samsum";
import "./Summary.css";
import { useEffect, useState } from "react";
import { useSummarize, useTopicAwareSummarize } from "../../hooks/APIHooks";
import CircularLoader from "../common/Loader/CircularLoader";
import useStore from "../../store/store";
import { CompareSummariesDialog } from "../common/CompareSummariesDialog/CompareSummariesDialog";
import { SummaryModel } from "./models/Summary";


interface SummaryProps {
  selectedDialogue: Samsum | null;
  selectedCompareTopic1: SummaryModel;
  selectedCompareTopic2: SummaryModel;
  setSelectedCompareTopic1: React.Dispatch<React.SetStateAction<SummaryModel>>
  setSelectedCompareTopic2: React.Dispatch<React.SetStateAction<SummaryModel>>
  userTopics:string[];
}

export function Summary({ selectedDialogue, selectedCompareTopic1, selectedCompareTopic2, setSelectedCompareTopic1, setSelectedCompareTopic2, userTopics}: SummaryProps) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const topicSelectTitle = "Topic";
  const { fetchSummary, summary, loading, error } = useSummarize([selectedDialogue ?? { id: "-1", summary: "", dialogue: "" }])
  const { fetchTopicAwareSummary, summaries: topicAwareSummaries, loading: topicAwareLoading, error: topicAwareError } = useTopicAwareSummarize(selectedDialogue ?? { id: "-1", summary: "", dialogue: "" },userTopics)
  const { searchQuery, isSummarize } = useStore();
  const [selectedSummaryTopic, setSelectedSummaryTopic] = useState("Default");
  const summaryTopicList = Object.keys(topicAwareSummaries ?? {});
  summaryTopicList.unshift("Default");
  const topicTitle = "Topic: ";
  const topic = searchQuery;
  const summaryTime = "Summarized: 3 min ago";
  const summaryTitle = "Summary";
  const buttonText = "Compare Summaries";

  const handleChange = (event: SelectChangeEvent) => {
    setSelectedSummaryTopic(event.target.value as string);
  };

  const openDialog = () => {
    setDialogOpen(true);
  };

  const closeDialog = () => {
    setDialogOpen(false);
  };

  useEffect(() => {
    if (isSummarize) {
      fetchSummary().then(() => {
        fetchTopicAwareSummary();
      });
    }
  }, [isSummarize])

  if (loading || topicAwareLoading) {
    return <CircularLoader />
  }
  if (error || topicAwareError) {
    return <div> {error}</div>
  }
  return <>
    <div className="summary-page">
      <div className="topic-area">
        <div className="topic-title">{topicTitle}</div>
        <div className="topic">{topic}</div>
      </div>
      <div className="summarize-time">{summaryTime}</div>
      <div className="dropdown-area">
        <FormControl sx={{ m: 1, width: 180 }}>
          <InputLabel id="demo-simple-select-label">{topicSelectTitle}</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={selectedSummaryTopic}
            label="Topic"
            onChange={handleChange}
          >
            {summaryTopicList.map((summaryTopic) => (
              <MenuItem value={summaryTopic}>{summaryTopic}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </div>
      <div className="summary-title">{summaryTitle}</div>
      <div className="summary">
        {selectedSummaryTopic !== "Default" ? (topicAwareSummaries ?? {})[selectedSummaryTopic]
          :
          summary?.[0].summary
        }
      </div>
      <div className="divider" />
      <div className="button-area">
        <button className="compare-summaries-button" onClick={openDialog}>{buttonText}</button>
      </div>
      <CompareSummariesDialog
        dialogOpen={dialogOpen}
        closeDialog={closeDialog}
        selectedCompareTopic1={selectedCompareTopic1}
        selectedCompareTopic2={selectedCompareTopic2}
        summaryTopicList={topicAwareSummaries ?? {}}
        setSelectedCompareTopic1={setSelectedCompareTopic1}
        setSelectedCompareTopic2={setSelectedCompareTopic2} />
    </div>
    <div className="divider" />
  </>;
}
