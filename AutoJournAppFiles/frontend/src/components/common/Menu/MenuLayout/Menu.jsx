import "./Menu.css";
import { useState } from "react";
import { MenuItem } from "../MenuItem/MenuItem";
import homeIcon from "../../../../assets/icons/home.png";
import searchIcon from "../../../../assets/icons/search.png";
import settingsIcon from "../../../../assets/icons/settings.png";
import { useNavigate } from "react-router-dom";

export function Menu() {
  const [selectedItem, setSelectedItem] = useState(null);
  const navigate = useNavigate();

  const handleMenuItemClick = (item, path) => {
    setSelectedItem(item);
    navigate(path);
  };

  return (
    <div className="menu-div">
      <MenuItem
        icon={homeIcon}
        text="Home"
        onClick={() => handleMenuItemClick("home", "/")}
        className={`menu-item ${selectedItem === "home" ? "menu-item-selected" : ""}`}
      />
      <MenuItem
        icon={searchIcon}
        text="Search"
        onClick={() => handleMenuItemClick("search", "/search")}
        className={`menu-item ${selectedItem === "search" ? "menu-item-selected" : ""}`}
      />
      <MenuItem
        icon={settingsIcon}
        text="Settings"
        onClick={() => handleMenuItemClick("settings", "/settings")}
        className={`menu-item ${selectedItem === "settings" ? "menu-item-selected" : ""}`}
      />
    </div>
  );
}
