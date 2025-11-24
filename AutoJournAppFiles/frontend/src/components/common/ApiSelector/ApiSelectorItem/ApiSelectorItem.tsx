import { MouseEventHandler } from "react";
import "./ApiSelectorItem.css";

interface ApiSelectorItemProps {
  text: string;
  isSelected: boolean;
  onClick: MouseEventHandler;
}

export function ApiSelectorItem({
  text,
  isSelected,
  onClick,
}: ApiSelectorItemProps) {
  return (
    <>
      <div
        className={`api-selector-text ${isSelected ? "selected" : ""}`}
        onClick={onClick}
      >
        {text}
      </div>
    </>
  );
}
