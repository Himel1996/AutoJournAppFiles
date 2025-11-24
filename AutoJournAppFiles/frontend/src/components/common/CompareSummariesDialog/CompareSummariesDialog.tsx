import {Dialog, DialogTitle, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent} from "@mui/material";
import {useNavigate} from "react-router-dom";
import { SummaryModel } from "../../Summary/models/Summary";
import { TopicAwareSummaryType } from "../../../hooks/APIHooks";


interface SummaryProps {
    dialogOpen:boolean;
    closeDialog: ()=>void;
    selectedCompareTopic1:SummaryModel;
    selectedCompareTopic2:SummaryModel;
    summaryTopicList:TopicAwareSummaryType;
    setSelectedCompareTopic1: React.Dispatch<React.SetStateAction<SummaryModel>>
    setSelectedCompareTopic2: React.Dispatch<React.SetStateAction<SummaryModel>>
}

export function CompareSummariesDialog({dialogOpen, closeDialog, selectedCompareTopic1,selectedCompareTopic2,summaryTopicList,setSelectedCompareTopic1,setSelectedCompareTopic2}:SummaryProps)
{
    const navigate = useNavigate();
    const topicSelectTitle = "Topic";

    const selectCompareTopic1 = (event: SelectChangeEvent) => {
        let selectedTopic = event.target.value as string;
        let selectedTopicValue = summaryTopicList[selectedTopic];
        setSelectedCompareTopic1({title: selectedTopic, content: selectedTopicValue});
    }
    const selectCompareTopic2 = (event: SelectChangeEvent) => {
        let selectedTopic = event.target.value as string;
        let selectedTopicValue = summaryTopicList[selectedTopic];
        setSelectedCompareTopic2({title: selectedTopic, content: selectedTopicValue});
    }

    const compareSummaries = () => {
        navigate('/compare-summaries');
    }

    const getSummaryTopicList = (selectedCompareTopic:string | undefined):string[] | undefined => {
        if(selectedCompareTopic==selectedCompareTopic1.title){
            return Object.keys(summaryTopicList).filter(topic => topic !== selectedCompareTopic2.title);
        }
        if(selectedCompareTopic==selectedCompareTopic2.title){
            return Object.keys(summaryTopicList).filter(topic => topic !== selectedCompareTopic1.title);
        }
    };

    return <>
        <Dialog open={dialogOpen} onClose={closeDialog}>
            <div className="compare-dialog">
                <DialogTitle>Compare Summaries</DialogTitle>
                <h3>Topic 1:</h3>
                <FormControl sx={{m: 1, width: 180}}>
                    <InputLabel id="demo-simple-select-label">{topicSelectTitle}</InputLabel>
                    <Select
                        labelId="demo-simple-select-label"
                        id={selectedCompareTopic1.title}
                        value={selectedCompareTopic1.title}
                        label="Topic"
                        onChange={selectCompareTopic1}
                    >
                        {getSummaryTopicList(selectedCompareTopic1.title)?.map((summaryTopic) => (
                            <MenuItem value={summaryTopic}>{summaryTopic}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <h3>Topic 2:</h3>
                <FormControl sx={{m: 1, width: 180}}>
                    <InputLabel id="demo-simple-select-label">{topicSelectTitle}</InputLabel>
                    <Select
                        labelId="demo-simple-select-label"
                        id={selectedCompareTopic2.title}
                        value={selectedCompareTopic2.title}
                        label="Topic"
                        onChange={selectCompareTopic2}
                    >
                        {getSummaryTopicList(selectedCompareTopic2.title)?.map((summaryTopic) => (
                            <MenuItem value={summaryTopic}>{summaryTopic}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <div>
                    <button className="compare-button" disabled={selectedCompareTopic1.title=='' || selectedCompareTopic2.title==''} onClick={compareSummaries}>Compare</button>
                </div>
            </div>
        </Dialog>
    </>
}
