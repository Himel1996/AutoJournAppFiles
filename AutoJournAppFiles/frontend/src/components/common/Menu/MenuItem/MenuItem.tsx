import { MouseEventHandler } from "react";
import "./MenuItem.css";

interface MenuItemProps {
  text: string;
  icon: string;
  onClick: MouseEventHandler;
  isSelected?: boolean; // Optional prop to indicate if the item is selected
}

export function MenuItem({ text, icon, onClick, isSelected }: MenuItemProps) {
  return (
    <div className={`item-container ${isSelected ? "item-selected" : ""}`} onClick={onClick}>
      <img loading="lazy" src={icon} className="item-icon" alt={`${text} icon`} />
      <div className="item-text">{text}</div>
    </div>
  );
}
