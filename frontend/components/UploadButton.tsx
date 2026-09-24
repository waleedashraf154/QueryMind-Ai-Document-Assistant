"use client";

import { useRef, useState } from "react";

export default function UploadButton({
  onFileSelected,
  compact = false,
}: {
  onFileSelected: (file: File) => void;
  compact?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  function handleFile(file: File | undefined) {
    if (!file) return;
    setFileName(file.name);
    onFileSelected(file);
  }

  if (compact) {
    return <>
      <button onClick={() => inputRef.current?.click()} className="grid h-10 w-10 shrink-0 place-items-center rounded-xl text-slate/45 transition hover:bg-parchment hover:text-signal" aria-label="Attach a document"><PlusIcon /></button>
      <input ref={inputRef} type="file" accept=".pdf,.docx,.pptx,.xlsx,.csv,.txt,.png,.jpg,.jpeg,.webp,.bmp" className="hidden" onChange={(e) => handleFile(e.target.files?.[0])} />
    </>;
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFile(e.dataTransfer.files?.[0]);
      }}
      onClick={() => inputRef.current?.click()}
      className={`flex cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors ${
        dragging
          ? "border-signal bg-signal/10 text-signal"
          : "border-slate/20 text-slate/60 hover:border-slate/40 hover:text-slate"
      }`}
    >
      <PaperclipIcon />
      <span className="truncate">{fileName ?? "Attach a document"}</span>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.pptx,.xlsx,.csv,.txt,.png,.jpg,.jpeg,.webp,.bmp"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
    </div>
  );
}

function PaperclipIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21.44 11.05l-9.19 9.19a5 5 0 01-7.07-7.07l9.19-9.19a3.5 3.5 0 014.95 4.95l-9.2 9.19a1.5 1.5 0 01-2.12-2.12l8.49-8.48" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function PlusIcon() {
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>;
}
