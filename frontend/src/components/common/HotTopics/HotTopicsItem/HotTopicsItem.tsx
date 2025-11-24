import "./HotTopicsItem.css";

interface HotTopicsItemProps {
  text: string;
}

export function HotTopicsItem({ text }: HotTopicsItemProps) {
  return (
    <>
      <div className="topic-item">{text}</div>
    </>
  );
}
