import { useEffect, useState } from "react";
import { Reddit, SocialAPI, Telegram } from "../api/SocialAPI";
import { Samsum } from "../backend-objects/Samsum";
import { ConversationSummary, TopicAwareSummary } from "../api/Summary";
import useStore from "../store/store";
import { APIConstants } from "../constants/APIConstants";
import { API_BASE_URL } from "../config";

const backendUrl = API_BASE_URL;

const teleAPI = new Telegram();
const redditAPI = new Reddit();

export const pickAPI = (api: APIConstants) => {
    switch (api) {
        case APIConstants.TELEGRAM:
            return teleAPI;
        case APIConstants.REDDIT:
            return redditAPI;
        default:
            return redditAPI; // Default to Reddit
    }
}

export const useFetchSearch = (apiConstant: APIConstants) => {
    const api = pickAPI(apiConstant);

    const { searchQuery, setConversations } = useStore();
    const [data, setData] = useState<[Samsum] | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            if (searchQuery === "" || searchQuery === null) {
                setLoading(false);
                setData(null);
                setError(null);
                return;
            }
            setLoading(true)
            try {
                const data = await api.getFeedData(searchQuery);
                setData(data);
                setConversations(data);
                setLoading(false);
            } catch (error) {
                setError("Error fetching data");
                setLoading(false);
            }finally {
                setLoading(false); // Stop loading
            }
        }

        fetchData();
    }, [searchQuery]);

    return { data, loading, error };
}


export interface SummarizeRequest {
    summaries: {
        [key: string]: string;
    };
    plot_type: string;
    dialogue: string;
    default_summary: string;
}

export const useFetchDeltaSummarize = (requestData: SummarizeRequest) => {
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch(backendUrl + '/delta-summarize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestData),
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const imageBlob = await response.blob();
                const imageObjectURL = URL.createObjectURL(imageBlob);
                setImageSrc(imageObjectURL);
                setLoading(false);
            } catch (error) {
                setError('Error fetching data');
                setLoading(false);
            }
        };

        fetchData();
    }, [requestData]);

    return { imageSrc, loading, error };
};


export const useSummarize = (dialogue: Samsum[]) => {
    const [summary, setSummary] = useState<[Samsum] | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const { setDefaultSummary } = useStore();
    const fetchSummary = async () => {
        setLoading(true);
        setError(null);
        try {
            const summaryAPI = new ConversationSummary(dialogue)
            const summaryRes = await summaryAPI.getSummary();
            setSummary(summaryRes);
            // Default summary saved in the store for delta summarization.
            setDefaultSummary(summaryRes[0]?.summary);
        } catch (err) {
            setError('Error fetching summary');
        } finally {
            setLoading(false);
        }
    };

    return { fetchSummary, summary, loading, error };
}

export interface TopicAwareSummaryType {
    [key: string]: string;
}


export const useTopicAwareSummarize = (dialogue: Samsum, userTopics?: string[]) => {
    const { conversations, setIsSummarize } = useStore();
    const [summaries, setSummaries] = useState<TopicAwareSummaryType | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const fetchTopicAwareSummary = async () => {
        setLoading(true);
        setError(null);
        try {
            const topicAwareSummaryAPI = new TopicAwareSummary(conversations, dialogue, 10, userTopics);
            const summaries = await topicAwareSummaryAPI.getSummary();
            setSummaries(summaries);
            // Set isSummarize false after summary is performed.
            setIsSummarize(false);
        } catch (err) {
            setError('Error fetching summary');
        } finally {
            setLoading(false);
        }
    };

    return { fetchTopicAwareSummary, summaries, loading, error };
}
