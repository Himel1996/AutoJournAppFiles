import "./SearchBar.css";
import searchIcon from "../../../assets/icons/search.png";
import { Input } from "@mui/material";
import { useState } from "react";

export function SearchBar({ setSearchQuery, setLoadingSearch}: { setSearchQuery: (query: string) => void; setLoadingSearch: (loading: boolean) => void; }) {
  const [query, setQuery] = useState("");

  const handleSearch = async () => {
    setLoadingSearch(true);; // Show loading screen
    try {
      //await new Promise((resolve) => setTimeout(resolve, 3000)); // Simulate loading delay
      setSearchQuery(query);
    } catch (error) {
      console.error("Error during search:", error);
    } finally {
      setLoadingSearch(false); // Hide loading screen
    }
  };

  return (
    <div className="search-bar-wrapper">
      <div className="search-bar-container">
        <img
          loading="lazy"
          src={searchIcon}
          className="search-icon"
          onClick={handleSearch}
          alt="Search Icon"
        />
        <Input
          className="search-bar-inner"
          placeholder="Search..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handleSearch}>Search</button>
      </div>
    </div>
  );
}
