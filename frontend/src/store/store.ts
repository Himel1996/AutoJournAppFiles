import { create } from 'zustand';
import { Samsum } from '../backend-objects/Samsum';

interface State {
    searchQuery: string;
    setSearchQuery: (searchQuery: string) => void;
    conversations: Samsum[];
    setConversations: (conversations: Samsum[]) => void;
    resetQuery: () => void;
    isSummarize: boolean;
    setIsSummarize: (isSummarize: boolean) => void;
    defaultSummary: string;
    setDefaultSummary: (defaultSummary: string) => void;
}

const useStore = create<State>((set) => ({
    searchQuery: "",
    setSearchQuery: (searchQuery) => set({ searchQuery: searchQuery }),
    conversations: [],
    setConversations: (conversations) => set({ conversations: conversations }),
    resetQuery: () => set({ searchQuery: "" }),
    isSummarize: false,
    setIsSummarize: (isSummarize) => set({ isSummarize: isSummarize }),
    defaultSummary: "",
    setDefaultSummary: (defaultSummary) => set({ defaultSummary: defaultSummary }),
}));

export default useStore;