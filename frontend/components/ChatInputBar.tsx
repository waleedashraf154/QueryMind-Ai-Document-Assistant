"use client";

import { useRef, useState, useEffect } from "react";
import { useUser } from "@clerk/nextjs";
import { getStoredDocuments, deleteStoredDocuments } from "@/lib/api";

/* ─── SVG icons ─────────────────────────────────────────────────── */
function PlusIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}
function UploadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
    </svg>
  );
}
function RecentIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="5" y="2" width="14" height="20" rx="2" />
      <path d="M9 6h6M9 10h6M9 14h4" />
    </svg>
  );
}
function ChevronRight() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M9 18l6-6-6-6" />
    </svg>
  );
}
function ChevronLeft() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M15 18l-6-6 6-6" />
    </svg>
  );
}
function MicIconSvg() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6" />
    </svg>
  );
}
function FileDocIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2.8H7a2 2 0 0 0-2 2v14.4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7.8Z" />
      <path d="M14 2.8v5h5" />
    </svg>
  );
}
function XIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

/* ─── Send button ────────────────────────────────────────────── */
function SendButton({ onClick, hasText }: { onClick: () => void; hasText: boolean }) {
  return (
    <button
      onClick={onClick}
      aria-label="Send message"
      style={{
        width: 36, height: 36, borderRadius: "50%",
        background: "linear-gradient(135deg, #1a73e8 0%, #4fc3f7 100%)",
        border: "none", cursor: "pointer",
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0, transition: "opacity 0.18s, transform 0.15s",
        opacity: hasText ? 1 : 0.55,
      }}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 19V5M5 12l7-7 7 7" />
      </svg>
    </button>
  );
}

/* ─── Attached file chip ─────────────────────────────────────── */
type DocStatus = "uploading" | "processing" | "ready" | "failed";

function StatusBadge({ status, progress }: { status: DocStatus; progress?: string }) {
  if (status === "ready") {
    return (
      <span style={{
        display: "inline-flex", alignItems: "center", gap: 3,
        fontSize: 11, fontWeight: 600, color: "#4ade80",
        background: "rgba(74,222,128,0.12)", borderRadius: 5,
        padding: "1px 6px",
      }}>
        ✓ Ready
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span style={{
        display: "inline-flex", alignItems: "center", gap: 3,
        fontSize: 11, fontWeight: 600, color: "#f87171",
        background: "rgba(248,113,113,0.12)", borderRadius: 5,
        padding: "1px 6px",
      }} title={progress || "Processing failed"}>
        ✗ Failed
      </span>
    );
  }
  // uploading or processing: show spinner
  const label = status === "uploading" ? "Uploading…" : (progress || "Processing…");
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      fontSize: 11, fontWeight: 500, color: "rgba(232,228,240,0.55)",
    }}>
      <span style={{
        display: "inline-block", width: 9, height: 9,
        border: "1.5px solid rgba(232,228,240,0.25)",
        borderTopColor: "#4fc3f7",
        borderRadius: "50%",
        animation: "qmSpin 0.7s linear infinite",
        flexShrink: 0,
      }} />
      {label}
    </span>
  );
}

function FileChip({
  name, status, progress, onRemove,
}: {
  name: string;
  status: DocStatus;
  progress?: string;
  onRemove: () => void;
}) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.10)",
      borderRadius: 8, padding: "4px 10px 4px 8px",
      fontSize: 13, color: "rgba(232,228,240,0.85)", fontWeight: 500, maxWidth: "100%",
    }}>
      <span style={{ color: "rgba(232,228,240,0.5)", flexShrink: 0 }}><FileDocIcon /></span>
      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 160 }}>{name}</span>
      <StatusBadge status={status} progress={progress} />
      <button
        onClick={onRemove}
        aria-label={`Remove ${name}`}
        style={{
          background: "none", border: "none", cursor: "pointer",
          color: "rgba(232,228,240,0.4)", padding: 0,
          display: "flex", alignItems: "center", flexShrink: 0, transition: "color 0.12s",
        }}
        onMouseEnter={e => (e.currentTarget.style.color = "rgba(232,228,240,0.85)")}
        onMouseLeave={e => (e.currentTarget.style.color = "rgba(232,228,240,0.4)")}
      >
        <XIcon />
      </button>
    </div>
  );
}

/* ─── Attached doc type ──────────────────────────────────────── */
export type AttachedDoc = {
  id: string;
  name: string;
  fileId?: string;               // backend-assigned file_id from async upload
  status: DocStatus;             // drives chip badge
  progress?: string;             // progress_message from backend status poll
};

/* ─── Recent files user-scoped localStorage helpers ─────────── */
const LEGACY_RECENT_KEY = "qm_recent_files";
const MAX_RECENT = 5;
const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024; // 20 MB

const ALLOWED_EXTENSIONS = [
  ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt",
  ".png", ".jpg", ".jpeg", ".webp", ".bmp"
];

function validateAttachment(file: File): { valid: boolean; error?: string } {
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return { valid: false, error: "File exceeds the 20MB upload limit." };
  }

  const name = file.name || "";
  const lastDot = name.lastIndexOf(".");
  const ext = lastDot !== -1 ? name.slice(lastDot).toLowerCase() : "";

  if (!ext || !ALLOWED_EXTENSIONS.includes(ext)) {
    return {
      valid: false,
      error: `Unsupported file type '${ext || "unknown"}'. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`,
    };
  }

  return { valid: true };
}

function getUserRecentKey(userEmail?: string | null): string | null {
  if (!userEmail) return null;
  const clean = userEmail.trim().toLowerCase();
  return clean ? `qm_recent_files_${clean}` : null;
}

function saveToUserRecent(userEmail: string | null | undefined, name: string) {
  const key = getUserRecentKey(userEmail);
  if (!key) return;
  try {
    const stored: string[] = JSON.parse(localStorage.getItem(key) ?? "[]");
    const updated = [name, ...stored.filter((n) => n !== name)].slice(0, MAX_RECENT);
    localStorage.setItem(key, JSON.stringify(updated));
  } catch { /* ignore */ }
}

function getUserRecent(userEmail: string | null | undefined): string[] {
  const key = getUserRecentKey(userEmail);
  if (!key) return [];
  try { return JSON.parse(localStorage.getItem(key) ?? "[]"); }
  catch { return []; }
}

/* ─── Main component ─────────────────────────────────────────── */
export default function ChatInputBar({
  onSend,
  onFileSelected,
  onRecallDoc,
  onVoice,
  listening = false,
  attachedDocs = [],
  onRemoveDoc,
  onDocsDeleted,
}: {
  onSend: (text: string) => void;
  onFileSelected: (file: File) => void;
  onRecallDoc?: (filename: string) => void;
  onVoice: () => void;
  listening?: boolean;
  attachedDocs?: AttachedDoc[];
  onRemoveDoc?: (id: string) => void;
  onDocsDeleted?: (filenames: string[]) => void;
}) {
  const { user, isSignedIn } = useUser();
  const userEmail = user?.primaryEmailAddress?.emailAddress?.toLowerCase().trim();

  const [input, setInput]             = useState("");
  const [menuOpen, setMenuOpen]       = useState(false);
  const [showRecent, setShowRecent]   = useState(false);
  const [recentFiles, setRecentFiles] = useState<string[]>([]);
  const [selectedRecent, setSelectedRecent] = useState<string[]>([]);
  const [deleting, setDeleting]       = useState(false);
  const [toast, setToast]             = useState("");
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const dragCounter = useRef(0);
  const menuRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const lastPasteTimeRef = useRef<number>(0);

  /* Clean up any legacy un-scoped localStorage key on mount */
  useEffect(() => {
    try {
      localStorage.removeItem(LEGACY_RECENT_KEY);
    } catch {}
  }, []);

  /* Reset recent files whenever active user session changes */
  useEffect(() => {
    if (!isSignedIn || !userEmail) {
      setRecentFiles([]);
      setSelectedRecent([]);
      setShowRecent(false);
      setMenuOpen(false);
    } else {
      const cached = getUserRecent(userEmail);
      setRecentFiles(cached);
    }
  }, [isSignedIn, userEmail]);

  /* Prevent browser from opening dropped files anywhere on the page */
  useEffect(() => {
    function handleWindowDragOver(e: DragEvent) {
      e.preventDefault();
    }
    function handleWindowDrop(e: DragEvent) {
      e.preventDefault();
    }
    window.addEventListener("dragover", handleWindowDragOver);
    window.addEventListener("drop", handleWindowDrop);
    return () => {
      window.removeEventListener("dragover", handleWindowDragOver);
      window.removeEventListener("drop", handleWindowDrop);
    };
  }, []);

  /* Load recent files for current user from user-scoped cache & server whenever menu opens */
  useEffect(() => {
    if (!isSignedIn || !userEmail) {
      setRecentFiles([]);
      return;
    }

    if (!menuOpen && !showRecent) return;

    // Fast initial display from current user's cache
    const cached = getUserRecent(userEmail);
    if (cached.length > 0) {
      setRecentFiles(cached);
    }

    getStoredDocuments()
      .then((docs) => {
        const serverNames = docs.map((d) => d.filename);
        setRecentFiles(serverNames);
        // Sync user-scoped cache with server list
        const key = getUserRecentKey(userEmail);
        if (key) {
          try {
            localStorage.setItem(key, JSON.stringify(serverNames.slice(0, MAX_RECENT)));
          } catch {}
        }
      })
      .catch(() => {
        /* Keep user-scoped local cache if offline or backend error */
      });
  }, [menuOpen, showRecent, isSignedIn, userEmail]);

  /* Close dropdown on outside click */
  useEffect(() => {
    function handle(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
        setShowRecent(false);
      }
    }
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, []);

  /* Auto-resize textarea */
  useEffect(() => {
    const ta = textRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 180)}px`;
  }, [input]);

  /* Auto-dismiss toast after 4 s */
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 4000);
    return () => clearTimeout(t);
  }, [toast]);

  /* ── Single unified attachment processor ─────────────────── */
  function processAttachment(file: File) {
    const validation = validateAttachment(file);
    if (!validation.valid) {
      setToast(validation.error || "Invalid file.");
      return;
    }

    saveToUserRecent(userEmail, file.name);
    onFileSelected(file);
  }

  /* ── Drag & Drop handlers ────────────────────────────────── */
  function handleDragEnter(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDraggingOver(true);
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "copy";
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setIsDraggingOver(false);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = 0;
    setIsDraggingOver(false);

    const files = e.dataTransfer.files;
    if (!files || files.length === 0) return;

    for (let i = 0; i < files.length; i++) {
      processAttachment(files[i]);
    }
  }

function generatePastedImageFilename(mimeType: string): string {
  const now = new Date();
  const YYYY = now.getFullYear();
  const MM = String(now.getMonth() + 1).padStart(2, "0");
  const DD = String(now.getDate()).padStart(2, "0");
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  const timestamp = `${YYYY}${MM}${DD}-${hh}${mm}${ss}`;

  let randomId = "";
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    randomId = crypto.randomUUID().replace(/-/g, "").slice(0, 6);
  } else {
    randomId = Math.random().toString(36).substring(2, 8);
  }

  let ext = "png";
  const normalized = (mimeType || "").toLowerCase();
  if (normalized.includes("jpeg") || normalized.includes("jpg")) ext = "jpg";
  else if (normalized.includes("webp")) ext = "webp";
  else if (normalized.includes("gif")) ext = "gif";
  else if (normalized.includes("bmp")) ext = "bmp";
  else if (normalized.includes("svg")) ext = "svg";

  return `pasted-image-${timestamp}-${randomId}.${ext}`;
}

  /* ── Clipboard paste handler (Ctrl+V image detection) ───── */
  function handlePaste(e: React.ClipboardEvent) {
    const clipboardData = e.clipboardData;
    if (!clipboardData) return;

    let imageBlob: Blob | null = null;
    let mimeType = "";

    // 1. Check clipboard items for image blob
    const items = clipboardData.items;
    if (items && items.length > 0) {
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.type.startsWith("image/")) {
          imageBlob = item.getAsFile();
          mimeType = item.type;
          break;
        }
      }
    }

    // 2. Fallback: check files array in clipboardData
    if (!imageBlob && clipboardData.files && clipboardData.files.length > 0) {
      for (let i = 0; i < clipboardData.files.length; i++) {
        const f = clipboardData.files[i];
        if (f.type.startsWith("image/")) {
          imageBlob = f;
          mimeType = f.type;
          break;
        }
      }
    }

    // If an image was detected in the clipboard
    if (imageBlob) {
      // Intercept so the image is NOT pasted as broken text/base64 into the message
      e.preventDefault();
      e.stopPropagation();

      // Debounce rapid duplicate paste events within 400ms
      const now = Date.now();
      if (now - lastPasteTimeRef.current < 400) {
        return;
      }
      lastPasteTimeRef.current = now;

      const filename = generatePastedImageFilename(mimeType);
      const file = new File([imageBlob], filename, { type: mimeType || "image/png" });
      processAttachment(file);
      return;
    }

    // Normal text paste continues undisturbed
  }

  /* ── handlers ────────────────────────────────────────────── */
  function handleRemoveDoc(id: string) {
    if (fileRef.current) {
      fileRef.current.value = "";
    }
    onRemoveDoc?.(id);
  }

  function send() {
    const text = input.trim();
    if (!text) return;
    if (fileRef.current) {
      fileRef.current.value = "";
    }
    onSend(text);
    setInput("");
  }

  function handleMenuItem(id: string) {
    if (id === "upload") {
      setMenuOpen(false);
      setShowRecent(false);
      fileRef.current?.click();
    } else if (id === "recent") {
      setShowRecent((v) => !v);
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (files && files.length > 0) {
      for (let i = 0; i < files.length; i++) {
        processAttachment(files[i]);
      }
    }
    e.target.value = "";
    setMenuOpen(false);
    setShowRecent(false);
  }

  function handleRecentClick(name: string) {
    setMenuOpen(false);
    setShowRecent(false);
    if (onRecallDoc) {
      onRecallDoc(name);
    }
  }

  function toggleSelectRecent(name: string) {
    setSelectedRecent((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
    );
  }

  function handleSelectRecentDocs() {
    if (selectedRecent.length === 0) return;
    for (const name of selectedRecent) {
      onRecallDoc?.(name);
    }
    setSelectedRecent([]);
    setMenuOpen(false);
    setShowRecent(false);
  }

  async function handleDeleteRecentDocs() {
    if (selectedRecent.length === 0 || deleting) return;
    setDeleting(true);
    const toDelete = [...selectedRecent];
    try {
      await deleteStoredDocuments(toDelete);
      setRecentFiles((prev) => prev.filter((n) => !toDelete.includes(n)));
      const key = getUserRecentKey(userEmail);
      if (key) {
        try {
          const stored: string[] = JSON.parse(localStorage.getItem(key) ?? "[]");
          localStorage.setItem(
            key,
            JSON.stringify(stored.filter((n) => !toDelete.includes(n)))
          );
        } catch {}
      }
      onDocsDeleted?.(toDelete);
      setSelectedRecent([]);
      setToast(`Deleted ${toDelete.length} document${toDelete.length > 1 ? "s" : ""}.`);
      getStoredDocuments()
        .then((docs) => {
          const serverNames = docs.map((d) => d.filename);
          setRecentFiles(serverNames);
          if (key) {
            try {
              localStorage.setItem(key, JSON.stringify(serverNames.slice(0, MAX_RECENT)));
            } catch {}
          }
        })
        .catch(() => {});
    } catch (err) {
      setToast(err instanceof Error ? err.message : "Failed to delete documents.");
    } finally {
      setDeleting(false);
    }
  }

  function handleVoice() {
    if (typeof window !== "undefined" && (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)) {
      setToast("Microphone access is not supported on this browser.");
      return;
    }
    onVoice();
  }

  const hasText = input.trim().length > 0;

  return (
    <>
      <style>{`
        .qm-bar {
          background: #15141e;
          border-radius: 18px;
          border: 1px solid rgba(255,255,255,0.07);
          padding: 12px 14px 10px;
          display: flex;
          flex-direction: column;
          gap: 8px;
          box-shadow: 0 6px 32px rgba(0,0,0,0.4);
          position: relative;
          transition: border-color 0.18s, background-color 0.18s, box-shadow 0.18s;
        }
        .qm-bar.qm-drag-over {
          border: 1.5px dashed #4fc3f7 !important;
          background: rgba(79, 195, 247, 0.08) !important;
          box-shadow: 0 0 24px rgba(79, 195, 247, 0.22), 0 6px 32px rgba(0,0,0,0.5) !important;
        }
        .qm-chips { display:flex; flex-wrap:wrap; gap:6px; padding-bottom:2px; }
        .qm-textarea {
          width:100%; background:transparent; border:none; outline:none;
          color:#e8e4f0; font-size:15px; line-height:1.55; resize:none;
          font-family:inherit; min-height:24px; max-height:180px;
        }
        .qm-textarea::placeholder { color: rgba(232,228,240,0.30); }
        .qm-row { display:flex; align-items:center; gap:8px; }
        .qm-plus {
          width:32px; height:32px; border-radius:9px;
          border:1.5px solid rgba(255,255,255,0.13);
          background:transparent; color:rgba(232,228,240,0.55);
          display:flex; align-items:center; justify-content:center;
          cursor:pointer; transition:background 0.15s,color 0.15s,border-color 0.15s;
          flex-shrink:0;
        }
        .qm-plus:hover { background:rgba(255,255,255,0.08); color:#e8e4f0; border-color:rgba(255,255,255,0.25); }
        .qm-mic {
          width:32px; height:32px; border-radius:50%; border:none;
          background:transparent; color:rgba(232,228,240,0.45);
          display:flex; align-items:center; justify-content:center;
          cursor:pointer; transition:color 0.15s,background 0.15s; flex-shrink:0;
        }
        .qm-mic:hover { color:#e8e4f0; background:rgba(255,255,255,0.07); }
        .qm-mic.qm-listening {
          color:#ef4444; background:rgba(239,68,68,0.12);
          animation: micPulse 1s ease-in-out infinite;
        }
        @keyframes micPulse {
          0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
          50%      { box-shadow: 0 0 0 6px rgba(239,68,68,0); }
        }
        .qm-wrap { position:relative; }
        .qm-menu {
          position:absolute; bottom:calc(100% + 8px); left:0;
          background:#1a1926; border:1px solid rgba(255,255,255,0.09);
          border-radius:13px; padding:5px; min-width:210px;
          box-shadow:0 16px 48px rgba(0,0,0,0.65); z-index:9999;
          animation:qmIn 0.14s ease;
        }
        .qm-submenu {
          position:absolute; bottom:0; left:calc(100% + 6px);
          background:#1a1926; border:1px solid rgba(255,255,255,0.09);
          border-radius:13px; padding:6px; min-width:290px; max-width:340px;
          box-shadow:0 16px 48px rgba(0,0,0,0.65); z-index:10000;
          animation:qmIn 0.14s ease;
        }
        @keyframes qmIn {
          from { opacity:0; transform:translateY(5px) scale(.97); }
          to   { opacity:1; transform:translateY(0)   scale(1); }
        }
        .qm-item {
          display:flex; align-items:center; gap:10px;
          padding:9px 11px; border-radius:9px; cursor:pointer;
          color:rgba(232,228,240,0.88); font-size:14px; font-weight:500;
          transition:background 0.12s,color 0.12s; user-select:none; position:relative;
        }
        .qm-item:hover { background:rgba(255,255,255,0.07); color:#fff; }
        .qm-item svg { color:rgba(232,228,240,0.5); flex-shrink:0; }
        .qm-item:hover svg { color:rgba(232,228,240,0.9); }
        .qm-arrow { margin-left:auto; color:rgba(232,228,240,0.28); }
        .qm-empty { padding:14px 12px; font-size:13px; color:rgba(232,228,240,0.35); text-align:center; }
        .qm-toast {
          position:absolute; bottom:calc(100% + 10px); left:0; right:0;
          background:#2a1a1a; border:1px solid rgba(239,68,68,0.3);
          border-radius:10px; padding:9px 13px; font-size:13px;
          color:rgba(239,180,180,0.9); z-index:9998;
          animation:qmIn 0.15s ease;
        }
        .qm-sub-label {
          padding:8px 12px 4px; font-size:11px; font-weight:700;
          text-transform:uppercase; letter-spacing:0.1em;
          color:rgba(232,228,240,0.3);
        }
        @keyframes qmSpin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>

      <div style={{ position: "relative", width: "100%" }}>
        {/* Toast notification */}
        {toast && <div className="qm-toast">{toast}</div>}

        <div
          className={`qm-bar${isDraggingOver ? " qm-drag-over" : ""}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onPaste={handlePaste}
        >
          {/* Drag-over drop overlay */}
          {isDraggingOver && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                borderRadius: 18,
                backgroundColor: "rgba(21, 20, 30, 0.88)",
                backdropFilter: "blur(4px)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                color: "#4fc3f7",
                fontSize: 14,
                fontWeight: 600,
                pointerEvents: "none",
                zIndex: 10,
                border: "1.5px dashed #4fc3f7",
              }}
            >
              <UploadIcon />
              <span>Drop document to attach</span>
            </div>
          )}

          {/* File chips */}
          {attachedDocs.length > 0 && (
            <div className="qm-chips">
              {attachedDocs.map((doc) => (
                <FileChip
                  key={doc.id}
                  name={doc.name}
                  status={doc.status}
                  progress={doc.progress}
                  onRemove={() => handleRemoveDoc(doc.id)}
                />
              ))}
            </div>
          )}

          {/* Textarea */}
          <textarea
            ref={textRef}
            className="qm-textarea"
            placeholder="Type / to use slash commands"
            value={input}
            rows={1}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
          />

          {/* Action row */}
          <div className="qm-row">
            {/* + button */}
            <div className="qm-wrap" ref={menuRef}>
              <button
                className="qm-plus"
                aria-label="Attach or add"
                aria-expanded={menuOpen}
                onClick={() => { setMenuOpen((v) => !v); setShowRecent(false); }}
              >
                <PlusIcon />
              </button>

              {menuOpen && (
                <div className="qm-menu" role="menu">
                  {/* Upload File */}
                  <div className="qm-item" role="menuitem" tabIndex={0}
                    onClick={() => handleMenuItem("upload")}
                    onKeyDown={(e) => e.key === "Enter" && handleMenuItem("upload")}
                  >
                    <UploadIcon /><span>Upload File</span>
                  </div>

                  {/* Recent Uploads with submenu */}
                  <div className="qm-item" role="menuitem" tabIndex={0}
                    onClick={() => handleMenuItem("recent")}
                    onKeyDown={(e) => e.key === "Enter" && handleMenuItem("recent")}
                    style={{ background: showRecent ? "rgba(255,255,255,0.07)" : undefined }}
                  >
                    <RecentIcon /><span>Recent Uploads</span>
                    <span className="qm-arrow">{showRecent ? <ChevronLeft /> : <ChevronRight />}</span>

                    {showRecent && (
                      <div className="qm-submenu" onClick={(e) => e.stopPropagation()}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 8px 8px", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                          <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "rgba(232,228,240,0.4)" }}>Recent uploads</span>
                          {recentFiles.length > 0 && (
                            <button
                              onClick={() => {
                                if (selectedRecent.length === recentFiles.length) {
                                  setSelectedRecent([]);
                                } else {
                                  setSelectedRecent([...recentFiles]);
                                }
                              }}
                              style={{ background: "none", border: "none", color: "#4fc3f7", fontSize: 11, fontWeight: 600, cursor: "pointer", padding: 0 }}
                            >
                              {selectedRecent.length === recentFiles.length ? "Deselect all" : "Select all"}
                            </button>
                          )}
                        </div>

                        {recentFiles.length === 0 ? (
                          <p className="qm-empty">No recent files yet.<br />Upload a file to get started.</p>
                        ) : (
                          <div style={{ maxHeight: 220, overflowY: "auto", padding: "4px 0", display: "flex", flexDirection: "column", gap: 2 }}>
                            {recentFiles.map((name) => {
                              const isChecked = selectedRecent.includes(name);
                              return (
                                <label
                                  key={name}
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 8,
                                    padding: "7px 9px",
                                    borderRadius: 8,
                                    cursor: "pointer",
                                    background: isChecked ? "rgba(79, 195, 247, 0.12)" : "transparent",
                                    color: isChecked ? "#fff" : "rgba(232,228,240,0.85)",
                                    fontSize: 13,
                                    fontWeight: 500,
                                    userSelect: "none",
                                    transition: "background 0.12s",
                                  }}
                                  onMouseEnter={(e) => {
                                    if (!isChecked) e.currentTarget.style.background = "rgba(255,255,255,0.05)";
                                  }}
                                  onMouseLeave={(e) => {
                                    if (!isChecked) e.currentTarget.style.background = "transparent";
                                  }}
                                >
                                  <input
                                    type="checkbox"
                                    checked={isChecked}
                                    onChange={() => toggleSelectRecent(name)}
                                    style={{ accentColor: "#4fc3f7", cursor: "pointer", width: 14, height: 14 }}
                                  />
                                  <FileDocIcon />
                                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }} title={name}>
                                    {name}
                                  </span>
                                </label>
                              );
                            })}
                          </div>
                        )}

                        {recentFiles.length > 0 && (
                          <div style={{ display: "flex", alignItems: "center", gap: 8, paddingTop: 6, marginTop: 4, borderTop: "1px solid rgba(255,255,255,0.08)" }}>
                            <button
                              onClick={handleSelectRecentDocs}
                              disabled={selectedRecent.length === 0}
                              style={{
                                flex: 1,
                                padding: "6px 12px",
                                borderRadius: 8,
                                border: "none",
                                background: selectedRecent.length > 0 ? "linear-gradient(135deg, #1a73e8 0%, #4fc3f7 100%)" : "rgba(255,255,255,0.08)",
                                color: selectedRecent.length > 0 ? "#fff" : "rgba(232,228,240,0.35)",
                                fontSize: 12,
                                fontWeight: 600,
                                cursor: selectedRecent.length > 0 ? "pointer" : "not-allowed",
                                transition: "opacity 0.15s",
                              }}
                            >
                              Select {selectedRecent.length > 0 ? `(${selectedRecent.length})` : ""}
                            </button>
                            <button
                              onClick={handleDeleteRecentDocs}
                              disabled={selectedRecent.length === 0 || deleting}
                              style={{
                                padding: "6px 12px",
                                borderRadius: 8,
                                border: "1px solid rgba(239,68,68,0.3)",
                                background: "rgba(239,68,68,0.12)",
                                color: selectedRecent.length > 0 ? "#f87171" : "rgba(239,180,180,0.35)",
                                fontSize: 12,
                                fontWeight: 600,
                                cursor: selectedRecent.length > 0 && !deleting ? "pointer" : "not-allowed",
                                transition: "background 0.15s, color 0.15s",
                              }}
                            >
                              {deleting ? "Deleting…" : "Delete"}
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Mic button */}
            <button
              className={`qm-mic${listening ? " qm-listening" : ""}`}
              aria-label={listening ? "Listening…" : "Voice input"}
              style={{ marginLeft: "auto" }}
              onClick={handleVoice}
            >
              <MicIconSvg />
            </button>

            {/* Send button */}
            <SendButton onClick={send} hasText={hasText} />
          </div>
        </div>

        {/* Hidden file input: multiple files allowed */}
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.docx,.pptx,.xlsx,.csv,.txt,.png,.jpg,.jpeg,.webp,.bmp"
          className="hidden"
          onChange={handleFileChange}
          multiple
        />
      </div>
    </>
  );
}
