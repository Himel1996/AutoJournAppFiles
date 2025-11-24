import { Samsum } from "../backend-objects/Samsum";

const backendUrl = "http://localhost:8787";

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
        let result = await fetch(`${backendUrl}${this.fetchEndpoint}?query=${query}`);
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
        let result = await fetch(`${backendUrl}${this.fetchEndpoint}?query=${query}`);
        let resultJson = await result.json();
        return resultJson.conversations.map((item: any) => new Samsum(item.id, item.summary, item.dialogue));
    }
}