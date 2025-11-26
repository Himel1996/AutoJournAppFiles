import { Samsum } from "../backend-objects/Samsum";
import { API_BASE_URL } from "../config";

const backendUrl = API_BASE_URL;

export interface SocialAPI {
    fetchEndpoint: string;
    getFeedData(query: string): Promise<[Samsum]>; 
}


export class Telegram implements SocialAPI {
    fetchEndpoint: string;

    constructor() {
        this.fetchEndpoint = "/search-telegram";
    }

    async getFeedData(query: string): Promise<[Samsum]> {
        let result = await fetch(`${backendUrl}${this.fetchEndpoint}?query=${encodeURIComponent(query)}`);
        let resultJson = await result.json();
        return resultJson.conversations.map((item: any) => new Samsum(item.id, item.summary, item.dialogue));
    }
}

export class Reddit implements SocialAPI {
    fetchEndpoint: string;
    
    constructor() {
        this.fetchEndpoint = "/search-reddit";
    }

    async getFeedData(query: string): Promise<[Samsum]> {
        let result = await fetch(`${backendUrl}${this.fetchEndpoint}?query=${encodeURIComponent(query)}`);
        let resultJson = await result.json();
        return resultJson.conversations.map((item: any) => new Samsum(item.id, item.summary, item.dialogue));
    }
}