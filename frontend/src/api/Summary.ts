import { Samsum } from "../backend-objects/Samsum";
import axios from 'axios';

const backendUrl = "http://72.62.44.22:8000";

abstract class Summary {
    conversations: Samsum[];

    constructor(conversations: Samsum[]) {
        this.conversations = conversations;
    }

    abstract getSummary(): Promise<any>;
}

export class ConversationSummary extends Summary {
    constructor(conversations: Samsum[]) {
        super(conversations);
    }

    async getSummary(): Promise<any> {
        let response = await axios.post(backendUrl + '/summarize', {
            conversations: this.conversations
        })

        return response.data.summaries;
    }
}

export class TopicAwareSummary extends Summary {
    dialogue: Samsum;
    numOfTopics: number;
    usertopics?: string[];

    constructor(conversations: Samsum[], dialogue: Samsum, numOfTopics: number, userTopics?: string[]) {
        super(conversations);
        this.dialogue = dialogue;
        this.numOfTopics = numOfTopics;
        this.usertopics = userTopics;
    }

    async getSummary(): Promise<any> {
        // Make a POST request to the backend API to get the summary
        let response = await axios.post(backendUrl + '/topic-aware-summarize', {
            conversations: this.conversations,
            dialogue: this.dialogue,
            num_topics: this.numOfTopics,
            user_topics: this.usertopics
        })

        return response.data.conv_summaries;
    }
}