import React, { useState } from "react";
import "./Popup.css";

interface TopicInputPopupProps {
  onClose: () => void;
  onSave: (selectedOption: string) => Promise<void>; // Adjusted for async operations
}

const TopicInputPopup: React.FC<TopicInputPopupProps> = ({ onClose, onSave }) => {
  const [selectedOption, setSelectedOption] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false); // State for loading screen

  const handleOptionChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedOption(event.target.value);
  };

  const handleSave = async () => {
    if (selectedOption.trim() !== "") {
      setIsLoading(true); // Show loading screen
      try {
        await onSave(selectedOption); // Save the selected option and perform async action
      } catch (error) {
        console.error("Error during save:", error);
        alert("An error occurred. Please try again.");
      } finally {
        setIsLoading(false); // Hide loading screen
        onClose(); // Close the popup after saving
      }
    } else {
      alert("Please select an option before saving.");
    }
  };

  return (
    <div className="popup-overlay">
      {isLoading ? (
        <div className="loading-screen">
          <div className="spinner"></div>
          <p>Loading, please wait...</p>
        </div>
      ) : (
        <div className="popup-content">
          <h2>Select Topic Modelling Method</h2>
          <div>
            <label>
              <input
                type="radio"
                name="topicModelling"
                value="Mistral"
                checked={selectedOption === "Mistral"}
                onChange={handleOptionChange}
              />
               Topic Modelling with GPT-4o
            </label>
          </div>
          <div className="popup-actions">
            <button onClick={handleSave}>Save</button>
            <button onClick={onClose}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TopicInputPopup;
