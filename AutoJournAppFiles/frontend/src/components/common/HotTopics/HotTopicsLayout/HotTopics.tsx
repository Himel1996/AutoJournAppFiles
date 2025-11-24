import { HotTopicsItem } from "../HotTopicsItem/HotTopicsItem";
import "./HotTopics.css";

export function HotTopics() {
  return (
    <>
      <div className="hot-topics-container">
        <h4>Hot Topics</h4>
        <HotTopicsItem text="Something" />
        <HotTopicsItem text="Something" />
        <HotTopicsItem text="Something" />
        <HotTopicsItem text="Something" />
        <HotTopicsItem text="Something" />
        <HotTopicsItem text="Something" />
      </div>
    </>
  );
}
