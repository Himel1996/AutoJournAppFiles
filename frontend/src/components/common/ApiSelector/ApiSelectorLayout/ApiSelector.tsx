import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiSelectorItem } from "../ApiSelectorItem/ApiSelectorItem";
import "./ApiSelector.css";
import { APIConstants } from "../../../../constants/APIConstants";

interface ApiSelectorProps {
  setSelectedAPI: (api: APIConstants) => void;
}

export function ApiSelector(props: ApiSelectorProps) {
  const [selectedAPI, setSelectedAPI] = useState<APIConstants | null>(null);
  const navigate = useNavigate();

  function selectAPI(api: APIConstants) {
    setSelectedAPI(api);
    props.setSelectedAPI(api);
    navigate("/search");
  }

  const getClassNames = (api: APIConstants) =>
    `api-selector-item ${selectedAPI === api ? "api-selector-item-selected" : ""}`;

  return (
    <div className="api-selector-container">
      <p className="api-selector-message">Please select one of these options to proceed with your search:</p>
      <div className="api-selector-box">
        <div
          className={getClassNames(APIConstants.TWITTER)}
          onClick={() => selectAPI(APIConstants.TWITTER)}
        >
          X (Twitter)
        </div>
        <div
          className={getClassNames(APIConstants.REDDIT)}
          onClick={() => selectAPI(APIConstants.REDDIT)}
        >
          Reddit
        </div>
        <div
          className={getClassNames(APIConstants.TELEGRAM)}
          onClick={() => selectAPI(APIConstants.TELEGRAM)}
        >
          Telegram
        </div>
      </div>
    </div>
  );
}
