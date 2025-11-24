import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./App.css";
import { Feed } from "./components/Feed/Feed";
import { Logo } from "./components/common/Logo/Logo";
import { Menu } from "./components/common/Menu/MenuLayout/Menu";
import { HotTopics } from "./components/common/HotTopics/HotTopicsLayout/HotTopics";
import { Summary } from "./components/Summary/Summary";
import { Settings } from "./components/Settings/Settings";
import { useState } from "react";
import { Samsum } from "./backend-objects/Samsum";
import { CompareSummaries } from "./components/CompareSummaries/CompareSummaries";
import TopicsPage from "./components/Topics/TopicsPage";
import NewsArticlePage from "./components/Topics/News/NewsArticlePage";
import EchoChamberPage from "./components/Topics/News/EchoChamberPage";
import ClaimVerificationPage from "./components/Topics/News/ClaimVerificationPage";

function App() {
  const [selectedDialogue, setSelectedDialogue] = useState<Samsum | null>(null);
  const [userTopics, setUserTopics] = useState<string[]>([]);
  const [selectedCompareTopic1, setSelectedCompareTopic1] = useState({ title: "", content: "" });
  const [selectedCompareTopic2, setSelectedCompareTopic2] = useState({ title: "", content: "" });

  return (
    <>
      <BrowserRouter>
        <div className="left-layout">
          <Logo />
          <Menu />
        </div>
        <div className="content">
          <Routes>
          <Route path="/" element={<Feed setSelectedDialogue={setSelectedDialogue} setUserTopics={setUserTopics} />} />
            <Route path="/feed" element={<Feed setSelectedDialogue={setSelectedDialogue} setUserTopics={setUserTopics}/>} />
            <Route path="/search" element={<Feed setSelectedDialogue={setSelectedDialogue} isSearch={true} setUserTopics={setUserTopics}/>} />
            <Route path="/summary" element={<Summary
              selectedDialogue={selectedDialogue}
              selectedCompareTopic1={selectedCompareTopic1}
              selectedCompareTopic2={selectedCompareTopic2}
              setSelectedCompareTopic1={setSelectedCompareTopic1}
              setSelectedCompareTopic2={setSelectedCompareTopic2} 
              userTopics={userTopics}
              />}
            />
            <Route path="/compare-summaries" element={<CompareSummaries
              selectedCompareTopic1={selectedCompareTopic1}
              selectedCompareTopic2={selectedCompareTopic2}
              selectedDialogue={selectedDialogue} />}
            />
            <Route path="/settings" element={<Settings />} />
             {/* Add the TopicsPage route */}
            <Route path="/topics" element={<TopicsPage />} />
            <Route path="/news-article" element={<NewsArticlePage />} />
            <Route path="/echo-chamber" element={<EchoChamberPage />} />
            <Route path="/claim-verification" element={<ClaimVerificationPage />} />
          </Routes>
        </div>
      </BrowserRouter>
    </>
  );
}

export default App;
