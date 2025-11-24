import { useState } from "react";
import { ApiSelector } from "../common/ApiSelector/ApiSelectorLayout/ApiSelector";
import "./Feed.css";
import { SearchBar } from "../common/SearchBar/SearchBar";
import { useNavigate } from "react-router-dom";
import { Samsum } from "../../backend-objects/Samsum";
import { useFetchSearch } from "../../hooks/APIHooks";
import CircularLoader from "../common/Loader/CircularLoader";
import useStore from "../../store/store";
import { APIConstants } from "../../constants/APIConstants";
import TopicInputPopup from "../common/TopicInputPopup/TopicInputPopup";

interface FeedProps {
  setSelectedDialogue: React.Dispatch<React.SetStateAction<Samsum | null>>;
  isSearch?: boolean;
  setUserTopics: React.Dispatch<React.SetStateAction<string[]>>;
}

Feed.defaultProps = {
  isSearch: false,
};

export function Feed({ setSelectedDialogue, isSearch, setUserTopics }: FeedProps) {
  const navigate = useNavigate();
  const { searchQuery, setSearchQuery, setIsSummarize, conversations } = useStore();
  const [selectedAPI, setSelectedAPI] = useState(APIConstants.REDDIT);
  const [selectedDialogueIndex, setselectedDialogueIndex] = useState(-1);
  const [loadingSearch, setLoadingSearch] = useState(false); // search loading state
  const { data: conversationResponse, loading, error } = useFetchSearch(selectedAPI);
  const [showPopup, setShowPopup] = useState<boolean>(false);
  const [selectedOption, setSelectedOption] = useState<string>("");

  const handlePopupClose = () => {
    setShowPopup(false);
  };

  //const handlePopupSave = (selectedOption: string) => {
  //  setSelectedOption(selectedOption);
  //  summarizeText(selectedOption);
  //};

  const handlePopupSave = async (selectedOption: string) => {
    setSelectedOption(selectedOption);
    console.log(`Selected option: ${selectedOption}`); // Debugging
    console.log(`Endpoint: /topics/${selectedOption}`);

    const requestBody = {
      conversations: conversations.map((conv) => conv.dialogue),
    };
    console.log("Request body:", requestBody);
    // Concatenate all dialogues into a single string for the original text
    const combinedText = conversations.map((conv) => conv.dialogue).join(" ");
  
    try {
      const response = await fetch(`http://127.0.0.1:8787/topics/${selectedOption}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          conversations: conversations.map((conv) => conv.dialogue),
          selectedModel: selectedOption
        }),
      });
  
      if (!response.ok) {
        throw new Error(`Failed to fetch topics: ${response.statusText}`);
      }
  
      const data = await response.json();
      console.log(`Topics received from ${selectedOption}:`, data);
      //navigate("/topics", { state: { topics: data.topics, keywords: data.keywords } });
      navigate("/topics", { state: { topics: data.topics, keywords: data.keywords, originalText: combinedText, topicsAndKeywords: data.topics_and_keywords } });
    } catch (error) {
      console.error(`Error fetching topics with ${selectedOption}:`, error);
    }
  };
  
  
  

  const title = "Topic: ";
  let source = `Using ${selectedAPI} API`;
  const buttonText = "Summarize Text";

  const summarizeText = (selectedOption: string) => {
    if (!conversationResponse) {
      return;
    }
    if (selectedDialogueIndex !== -1) {
      setSelectedDialogue(conversationResponse[selectedDialogueIndex]);
      setUserTopics([selectedOption]); // Save the selected option
      setIsSummarize(true);
      navigate("/summary");
    }
  };

  const selectDialogue = (index: number) => {
    if (selectedDialogueIndex === index) {
      setselectedDialogueIndex(-1);
    } else {
      setselectedDialogueIndex(index);
    }
  };

  const handleSearch = async (query: string) => {
    setLoadingSearch(true); // Start loading animation
    setSearchQuery(query); // Assuming this triggers the data fetch
    //setLoadingSearch(false); // Stop loading animation after search
  };

  if (error) {
    return <div> {error}</div>;
  }
  if (loading || loadingSearch) {
    return (
      <div className="loading-screen">
      <div className="spinner"></div>
      <p>Loading conversations, please wait...</p>
    </div>
    );
  }
  return (
    <div className="feed-wrapper"> {/* Add wrapper div for max width */}
      <div className="api-selector">
        <ApiSelector setSelectedAPI={setSelectedAPI} />
        {isSearch ? <SearchBar setSearchQuery={handleSearch} setLoadingSearch={setLoadingSearch} /> : <></>}
      </div>
      <div className="dialogues-area">
        <div className="topic-area">
          <div className="topic-title">{title}</div>
          <div className="topic">{searchQuery}</div>
        </div>
        <div className="fetch-time">{source}</div>
        <div className="dialogue-list">
          {(conversations ?? []).map((samsum, index) => (
            <div
              className={`dialogue ${selectedDialogueIndex === index ? "selected" : ""}`}
              onClick={() => selectDialogue(index)}
              key={samsum.id}
            >
              {samsum.dialogue}
            </div>
          ))}
        </div>
        <div className="divider" />
        <div className="divider" />
        <button
          className="summarize-button"
          onClick={() => setShowPopup(true)}
          disabled={selectedDialogueIndex === -1}
        >
          {buttonText}
        </button>
        {showPopup && (
          <TopicInputPopup onClose={handlePopupClose} onSave={handlePopupSave} />
        )}
      </div>
    </div>
  );
}
