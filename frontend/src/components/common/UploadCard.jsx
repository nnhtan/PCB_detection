import { useRef, useState } from "react";
import { ImagePlus, UploadCloud } from "lucide-react";

export default function UploadCard({
  title = "Tải ảnh PCB",
  description = "Kéo thả ảnh hoặc nhấn để chọn tệp",
  onSelect,
  accept = "image/*",
  preview,
  compact = false,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const select = (file) => {
    if (file) onSelect(file);
  };
  return (
    <div
      className={`upload-card ${dragging ? "is-dragging" : ""} ${compact ? "compact" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        select(event.dataTransfer.files[0]);
      }}
    >
      <input
        ref={inputRef}
        hidden
        type="file"
        accept={accept}
        onChange={(event) => select(event.target.files[0])}
      />
      {preview ? (
        <button className="preview-button" type="button" onClick={() => inputRef.current?.click()}>
          <img src={preview} alt="Ảnh PCB đã chọn" />
          <span><ImagePlus size={15} /> Thay ảnh</span>
        </button>
      ) : (
        <button className="upload-content" type="button" onClick={() => inputRef.current?.click()}>
          <span className="upload-icon"><UploadCloud size={22} /></span>
          <strong>{title}</strong>
          <span>{description}</span>
          <span className="upload-browse">Chọn ảnh từ thiết bị</span>
        </button>
      )}
    </div>
  );
}
