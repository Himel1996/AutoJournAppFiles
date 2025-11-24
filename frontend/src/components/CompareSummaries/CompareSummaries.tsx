import { FormControlLabel, Switch } from "@mui/material";
import "./CompareSummaries.css"
import { useState } from "react";
import { SummarizeRequest, useFetchDeltaSummarize } from "../../hooks/APIHooks";
import { SummaryModel } from "../Summary/models/Summary";
import { Samsum } from "../../backend-objects/Samsum";
import useStore from "../../store/store";

interface CompareSummariesProps {
    selectedCompareTopic1: SummaryModel;
    selectedCompareTopic2: SummaryModel;
    selectedDialogue: Samsum | null;
}


export function CompareSummaries({ selectedCompareTopic1, selectedCompareTopic2, selectedDialogue }: CompareSummariesProps) {
    const { defaultSummary } = useStore();
    const graphTypes = ["Scatter Plot", "Cosine Similarity Matrix"];
    const [switchLabel, setSwitchLabel] = useState(graphTypes[0]);
    const [checked, setChecked] = useState(false);
    const [requestData, setRequestData] = useState({
        summaries: {
            [selectedCompareTopic1.title]: selectedCompareTopic1.content,
            [selectedCompareTopic2.title]: selectedCompareTopic2.content
        },
        plot_type: '2d_scatter_plot', // scatter plot
        default_summary: defaultSummary ?? "",
        dialogue: selectedDialogue?.dialogue ?? ""
    });

    const { imageSrc, loading, error } = useFetchDeltaSummarize(requestData);

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setChecked(event.target.checked);

        let data: SummarizeRequest = {
            summaries: {},
            plot_type: '',
            default_summary: '',
            dialogue: ''
        }

        data.summaries[selectedCompareTopic1.title] = selectedCompareTopic1.content;
        data.summaries[selectedCompareTopic2.title] = selectedCompareTopic2.content;

        if (checked) {
            setSwitchLabel(graphTypes[0]);
            data.plot_type = '2d_scatter_plot';
            data.default_summary = defaultSummary ?? "";
            data.dialogue = selectedDialogue?.dialogue ?? "";
            setRequestData(data);
        }
        else {
            setSwitchLabel(graphTypes[1]);
            data.plot_type = 'cosine_similarity';
            setRequestData(data);
        }
    };




    return <>
        <div className="compare-summary-page">
            <div className="title-area">
                Compare Summaries: {selectedCompareTopic1.title} and {selectedCompareTopic2.title}
            </div>
            <FormControlLabel
                control={
                    <Switch checked={checked} onChange={handleChange} defaultChecked color="default" />
                }
                label={switchLabel}
            />
            <div>
                {imageSrc && <img src={imageSrc} style={{ maxWidth: '600px', height: 'auto' }} />}
            </div>
        </div>
    </>
}