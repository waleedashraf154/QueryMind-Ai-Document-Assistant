"use client";

import { useEffect, useMemo, useState, useRef, useCallback } from "react";
import { UserButton, useUser, useClerk } from "@clerk/nextjs";
import Brand from "@/components/Brand";
import ChatInputBar, { AttachedDoc } from "@/components/ChatInputBar";
import ChatList, { Chat } from "@/components/ChatList";
import { ChatIcon, FileIcon, PanelIcon, PinIcon, PlusIcon, SearchIcon, SparkleIcon } from "@/components/Icons";
import FaceIdModal from "@/components/FaceIdModal";
import VoiceRecorderModal from "@/components/VoiceRecorderModal";
import MessageBubble, { Citation, Message } from "@/components/MessageBubble";
import ProcessingIndicator from "@/components/ProcessingIndicator";
import DocumentViewer from "@/components/DocumentViewer";
import { askQuestion, createChat, deleteChat, renameChat, getChats, uploadDocument, uploadDocumentAsync, getIngestionStatus, recallDocument, getChatSources, getStoredDocuments, deleteStoredDocuments, type StoredDoc, type ChatRecord, type Source } from "@/lib/api";

function toMessages(record?: ChatRecord): Message[] {
  return (record?.messages ?? []).map((m, index) => ({
    id: `m-${index}-${m.role}`,
    role: m.role,
    content: m.content,
  }));
}

// ─────────────────────────────────────────────────────────────
// How far from the bottom (px) before we consider the user
// to be "at the bottom" and hide the scroll-to-bottom button.
// ─────────────────────────────────────────────────────────────
const BOTTOM_THRESHOLD = 80;

export default function ChatPage() {
  const { user, isLoaded, isSignedIn } = useUser();
  const { openUserProfile, signOut } = useClerk();
  const [chats, setChats] = useState<Chat[]>([]);
  const [records, setRecords] = useState<Record<string, ChatRecord>>({});
  const [active, setActive] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [attachedDocs, setAttachedDocs] = useState<AttachedDoc[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [search, setSearch] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const [rightPanelView, setRightPanelView] = useState<"sources" | "preview">("sources");
  const [previewDoc, setPreviewDoc] = useState<{ filename: string; snippet?: string } | null>(null);
  const [voiceOpen, setVoiceOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const [citation, setCitation] = useState<Citation | null>(null);

  /** Explicitly closes document preview and returns active right-panel view to "sources". */
  const handleClosePreview = useCallback(() => {
    setRightPanelView("sources");
    setPreviewDoc(null);
    setCitation(null);
  }, []);

  /** Actively opens document preview in the right sidebar. */
  const handleOpenPreview = useCallback((doc: { filename: string; snippet?: string }, cit?: Citation) => {
    if (cit) setCitation(cit);
    setPreviewDoc(doc);
    setRightPanelView("preview");
    setRightOpen(true);
  }, []);

  /** Toggles the right sidebar visibility. */
  const handleToggleRightSidebar = useCallback(() => {
    setRightOpen((prev) => !prev);
  }, []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [faceModalOpen, setFaceModalOpen] = useState(false);
  const [collapsedFlyout, setCollapsedFlyout] = useState<"search" | "recent-uploads" | "pinned" | "chats" | null>(null);
  const [storedDocs, setStoredDocs] = useState<StoredDoc[]>([]);
  const [selectedStoredDocs, setSelectedStoredDocs] = useState<string[]>([]);
  const [loadingStoredDocs, setLoadingStoredDocs] = useState(false);
  const [deletingStoredDocs, setDeletingStoredDocs] = useState(false);

  // ── Scroll refs ──────────────────────────────────────────────
  // scrollContainerRef → the <div> that has overflow-y-auto (the actual scrolling element)
  // bottomSentinelRef  → an empty div at the very end of the message list
  // showScrollBtn      → whether the "scroll to bottom" FAB is visible
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomSentinelRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  // ── Scroll helpers ───────────────────────────────────────────
  /** Returns true when the scroll container is within BOTTOM_THRESHOLD px of the bottom. */
  const isNearBottom = useCallback((): boolean => {
    const el = scrollContainerRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_THRESHOLD;
  }, []);

  /** Scroll the container to the very bottom. */
  const scrollToBottom = useCallback((smooth = true) => {
    const el = scrollContainerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "instant" });
  }, []);

  // ── Scroll position tracking ─────────────────────────────────
  // Listen to the scroll container and show/hide the FAB.
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;

    function handleScroll() {
      setShowScrollBtn(!isNearBottom());
    }

    el.addEventListener("scroll", handleScroll, { passive: true });
    return () => el.removeEventListener("scroll", handleScroll);
  }, [isNearBottom]);

  // ── Auto-scroll on new messages ──────────────────────────────
  // Only auto-scroll when the user is already near the bottom
  // (following the conversation), OR when `busy` flips to false
  // (assistant just finished responding).
  // We do NOT auto-scroll if the user has scrolled far up to read history.
  const prevBusy = useRef(false);
  useEffect(() => {
    const justFinished = prevBusy.current && !busy;
    const justStarted = !prevBusy.current && busy;
    prevBusy.current = busy;

    if (isNearBottom() || justFinished || justStarted) {
      // Use requestAnimationFrame so the DOM has rendered the new message.
      requestAnimationFrame(() => scrollToBottom(true));
    }
  }, [messages, busy, isNearBottom, scrollToBottom]);

  // ── Switch chat → jump to bottom instantly ───────────────────
  useEffect(() => {
    // Use instant jump (no animation) when changing chats so it feels snappy.
    requestAnimationFrame(() => scrollToBottom(false));
    // Also reset the FAB visibility.
    setShowScrollBtn(false);
  // We only want to re-run this when the active chat changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  // ─────────────────────────────────────────────────────────────

  async function refreshChats(preferredId?: string) {
    const data = await getChats();
    setRecords(data);
    const next = Object.entries(data).map(([id, record]) => ({
      id,
      title: record.title || "New Chat",
    }));
    setChats(next);
    const nextActive = preferredId && data[preferredId] ? preferredId : active && data[active] ? active : next[0]?.id ?? null;
    setActive(nextActive);
    setMessages(toMessages(nextActive ? data[nextActive] : undefined));
  }

  // Account isolation: Reset and reload all state whenever Clerk user session changes
  useEffect(() => {
    if (!isLoaded) return;

    if (!isSignedIn || !user) {
      setChats([]);
      setRecords({});
      setActive(null);
      setMessages([]);
      setAttachedDocs([]);
      setSources([]);
      handleClosePreview();
      setStoredDocs([]);
      setSelectedStoredDocs([]);
      setError("");
      return;
    }

    // Instantly wipe prior user's state from memory
    setChats([]);
    setRecords({});
    setActive(null);
    setMessages([]);
    setAttachedDocs([]);
    setSources([]);
    handleClosePreview();
    setStoredDocs([]);
    setSelectedStoredDocs([]);
    setError("");

    // Load data strictly for the currently authenticated user
    void refreshChats().catch((e) => setError(e instanceof Error ? e.message : "Failed to load chats."));
    void getStoredDocuments()
      .then((docs) => setStoredDocs(docs))
      .catch(() => setStoredDocs([]));
  }, [isLoaded, isSignedIn, user?.id]);

  const recordsRef = useRef(records);
  recordsRef.current = records;
  const prevActiveRef = useRef<string | null>(null);

  // Update messages, sources, and attached docs whenever active chat changes
  useEffect(() => {
    if (prevActiveRef.current === active) return;
    const prev = prevActiveRef.current;
    prevActiveRef.current = active;

    // Avoid wiping local in-flight chat state when transitioning from null to a newly created active chat
    if (prev === null && active) {
      void getChatSources(active)
        .then((chatSources: Source[]) => {
          setSources(chatSources);
        })
        .catch(() => {
          setSources([]);
        });
      return;
    }

    if (active) {
      setMessages(toMessages(recordsRef.current[active]));
      setCitation(null);
      setAttachedDocs([]);

      void getChatSources(active)
        .then((chatSources: Source[]) => {
          setSources(chatSources);
        })
        .catch(() => {
          setSources([]);
        });
    } else {
      setMessages([]);
      setSources([]);
      setCitation(null);
      setAttachedDocs([]);
    }
  }, [active]);

  const visible = useMemo(
    () => chats.filter((chat) => chat.title.toLowerCase().includes(search.toLowerCase())),
    [chats, search],
  );
  const pinned = visible.filter((chat) => (chat as Chat & { pinned?: boolean }).pinned);
  const recent = visible.filter((chat) => !(chat as Chat & { pinned?: boolean }).pinned);

  useEffect(() => {
    if (leftOpen) {
      setCollapsedFlyout(null);
    }
  }, [leftOpen]);

  const openRecentUploadsFlyout = useCallback(async () => {
    setCollapsedFlyout("recent-uploads");
    setLoadingStoredDocs(true);
    try {
      const docs = await getStoredDocuments();
      setStoredDocs(docs);
    } catch {
      setStoredDocs([]);
    } finally {
      setLoadingStoredDocs(false);
    }
  }, []);

  const toggleFlyout = (type: "search" | "recent-uploads" | "pinned" | "chats") => {
    if (collapsedFlyout === type) {
      setCollapsedFlyout(null);
    } else {
      if (type === "recent-uploads") {
        void openRecentUploadsFlyout();
      } else {
        setCollapsedFlyout(type);
      }
    }
  };

  const handleSelectRecentFromFlyout = () => {
    if (selectedStoredDocs.length === 0) return;
    for (const name of selectedStoredDocs) {
      void handleRecallDoc(name);
    }
    setSelectedStoredDocs([]);
    setCollapsedFlyout(null);
  };

  const handleDeleteRecentFromFlyout = async () => {
    if (selectedStoredDocs.length === 0 || deletingStoredDocs) return;
    setDeletingStoredDocs(true);
    try {
      const res = await deleteStoredDocuments(selectedStoredDocs);
      const deletedNames = res.deleted ?? selectedStoredDocs;
      setStoredDocs((prev) => prev.filter((d) => !deletedNames.includes(d.filename)));
      setAttachedDocs((prev) => prev.filter((d) => !deletedNames.includes(d.name)));
      if (previewDoc && deletedNames.includes(previewDoc.filename)) {
        handleClosePreview();
      }
      setSelectedStoredDocs([]);
    } catch {
      // ignore
    } finally {
      setDeletingStoredDocs(false);
    }
  };

  function pin(id: string) {
    setChats((value) => value.map((chat) => chat.id === id ? { ...chat, pinned: !chat.pinned } : chat));
  }

  async function removeChat(id: string) {
    try {
      await deleteChat(id);
      const next = chats.filter((c) => c.id !== id);
      setChats(next);
      const nextRecords = { ...records };
      delete nextRecords[id];
      setRecords(nextRecords);
      if (active === id) {
        const nextId = next[0]?.id ?? null;
        setActive(nextId);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete chat.");
    }
  }

  async function handleRenameChat(id: string, newTitle: string) {
    const cleanTitle = newTitle.trim();
    if (!cleanTitle) return;

    // Optimistically update local state immediately
    setChats((prev) => prev.map((c) => (c.id === id ? { ...c, title: cleanTitle } : c)));
    setRecords((prev) => {
      if (!prev[id]) return prev;
      return { ...prev, [id]: { ...prev[id], title: cleanTitle } };
    });

    try {
      await renameChat(id, cleanTitle);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to rename chat.");
      void refreshChats();
    }
  }

  async function addChat() {
    const id = crypto.randomUUID();
    try {
      await createChat(id);
      setRecords((value) => ({ ...value, [id]: { title: "New Chat", messages: [] } }));
      setChats((value) => [{ id, title: "New Chat" }, ...value]);
      setActive(id);
      setMessages([]);
      setSources([]);
      setAttachedDocs([]);
      handleClosePreview();
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create chat.");
    }
  }

  async function ensureActiveChat() {
    if (active) return active;
    const id = crypto.randomUUID();
    await createChat(id);
    setRecords((value) => ({ ...value, [id]: { title: "New Chat", messages: [] } }));
    setChats((value) => [{ id, title: "New Chat" }, ...value]);
    setActive(id);
    return id;
  }

  async function handleSend(text: string) {
    if (busy) return;

    // Guard: if any attached document is still being ingested, block send and inform the user
    const stillProcessing = attachedDocs.some(
      (d) => d.status === "uploading" || d.status === "processing"
    );
    if (stillProcessing) {
      setError(
        "One or more documents are still being prepared. Please wait a moment and try again."
      );
      return;
    }

    setBusy(true);
    setError("");
    try {
      const chatId = await ensureActiveChat();
      setMessages((value) => [...value, { id: crypto.randomUUID(), role: "user", content: text }]);
      // Immediately jump to bottom after adding the user message.
      requestAnimationFrame(() => scrollToBottom(true));

      const result = await askQuestion(chatId, text);
      setMessages((value) => [...value, { id: crypto.randomUUID(), role: "assistant", content: result.answer }]);

      // Successfully sent message: clear all composer attachments
      setAttachedDocs(() => []);

      // Keep right panel sources in sync for this specific chat
      void getChatSources(chatId).then(setSources).catch(() => {});

      // If backend generated a new title, immediately update chat state and sidebar without page reload
      if (result.chat_title) {
        const newTitle = result.chat_title;
        setChats((prev) => prev.map((c) => (c.id === chatId ? { ...c, title: newTitle } : c)));
        setRecords((prev) => ({
          ...prev,
          [chatId]: {
            ...(prev[chatId] ?? { messages: [] }),
            title: newTitle,
            messages: [
              ...(prev[chatId]?.messages ?? []),
              { role: "user", content: text },
              { role: "assistant", content: result.answer },
            ],
          },
        }));
      } else {
        setRecords((value) => ({
          ...value,
          [chatId]: {
            ...(value[chatId] ?? { title: "New Chat", messages: [] }),
            messages: [
              ...(value[chatId]?.messages ?? []),
              { role: "user", content: text },
              { role: "assistant", content: result.answer },
            ],
          },
        }));
      }
    } catch (e) {
      // Preserve attachment in composer on error so user can retry safely
      setError(e instanceof Error ? e.message : "Unable to connect to QueryMind.");
    } finally {
      setBusy(false);
    }
  }

  /**
   * Polls the backend ingestion status for a single file chip every 2s.
   * Updates the chip badge as status changes: queued -> processing -> ready/failed.
   * On 'ready', refreshes the Sources panel for the active chat.
   * Runs independently per file — does NOT affect chatBusy/busy state.
   */
  async function pollIngestionStatus(chipId: string, fileId: string) {
    const POLL_INTERVAL_MS = 2000;
    const MAX_POLLS = 90; // 3 minutes max before giving up
    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise<void>((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      try {
        const result = await getIngestionStatus(fileId);
        setAttachedDocs((prev) =>
          prev.map((d) =>
            d.id === chipId
              ? {
                  ...d,
                  status: result.status as AttachedDoc["status"],
                  progress: result.progress_message,
                }
              : d
          )
        );
        if (result.status === "ready") {
          // Refresh Sources panel so the user can see the new document
          if (active) {
            void getChatSources(active).then(setSources).catch(() => {});
          }
          // Refresh stored docs list
          void getStoredDocuments().then(setStoredDocs).catch(() => {});
          break;
        }
        if (result.status === "failed" || result.status === "unknown") {
          break;
        }
      } catch {
        // Transient network error — keep polling
      }
    }
  }

  async function handleFile(file: File) {
    // NOTE: We do NOT set setBusy(true) here.
    // File upload and ingestion run in the background independently of chat.
    setError("");
    const chipId = crypto.randomUUID();

    // 1. Show chip immediately with "uploading" status
    setAttachedDocs((prev) => [
      ...prev,
      { id: chipId, name: file.name, status: "uploading" },
    ]);

    try {
      const chatId = await ensureActiveChat();

      // 2. POST /api/upload/async — returns in <500ms with file_id
      const { file_id } = await uploadDocumentAsync(chatId, file);

      // 3. Update chip: file is saved, background ingestion has started
      setAttachedDocs((prev) =>
        prev.map((d) =>
          d.id === chipId
            ? { ...d, fileId: file_id, status: "processing", progress: "Queued for processing…" }
            : d
        )
      );

      // 4. Start polling — this runs independently and does not await
      void pollIngestionStatus(chipId, file_id);

      // Refresh stored docs list immediately so sidebar shows the new file
      void getStoredDocuments().then(setStoredDocs).catch(() => {});
    } catch (e) {
      // Mark chip as failed so the user knows something went wrong
      setAttachedDocs((prev) =>
        prev.map((d) =>
          d.id === chipId
            ? { ...d, status: "failed", progress: e instanceof Error ? e.message : "Upload failed" }
            : d
        )
      );
      setError(e instanceof Error ? e.message : "Upload failed.");
    }
  }

  async function handleRecallDoc(filename: string) {
    // Recall is fast (checks Chroma cache first), so we set busy=true briefly
    // but still use the status chip to communicate state.
    setBusy(true);
    setError("");
    const chipId = crypto.randomUUID();
    setAttachedDocs((prev) => [
      ...prev.filter((d) => d.name !== filename),
      { id: chipId, name: filename, status: "processing", progress: "Loading document…" },
    ]);
    try {
      const chatId = await ensureActiveChat();
      await recallDocument(chatId, filename);

      // Mark chip as ready immediately (recall has its own fast path via Chroma)
      setAttachedDocs((prev) =>
        prev.map((d) => (d.id === chipId ? { ...d, status: "ready", progress: "" } : d))
      );

      // Refresh Sources & Context for this chat from backend immediately
      const chatSources = await getChatSources(chatId);
      setSources(chatSources);
    } catch (e) {
      setAttachedDocs((prev) =>
        prev.map((d) =>
          d.id === chipId
            ? { ...d, status: "failed", progress: e instanceof Error ? e.message : "Failed to recall" }
            : d
        )
      );
      setError(e instanceof Error ? e.message : `Failed to recall '${filename}'.`);
    } finally {
      setBusy(false);
    }
  }

  function voice() {
    setVoiceOpen(true);
  }

  return (
    <>
    {/* Root container — fills the FULL viewport with no outer gaps */}
    <div className="flex h-screen w-screen overflow-hidden bg-parchment text-slate">
      {/* ── Left Sidebar: Smooth width & opacity transition between expanded (280px) and collapsed (64px) ── */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 flex h-screen flex-col border-r border-white/10 bg-ink text-parchment shadow-2xl lg:static lg:shadow-none lg:shrink-0 transition-[width] duration-300 ease-[cubic-bezier(0.2,0,0,1)] overflow-hidden ${
          leftOpen ? "w-[280px]" : "w-16"
        }`}
      >
        {/* Expanded Content: fixed 280px inner width to prevent text wrapping/jumping during animation */}
        <div
          className={`absolute inset-0 flex h-full w-[280px] flex-col p-4 transition-all duration-300 ease-[cubic-bezier(0.2,0,0,1)] ${
            leftOpen
              ? "opacity-100 translate-x-0 pointer-events-auto"
              : "opacity-0 -translate-x-6 pointer-events-none"
          }`}
        >
          <div className="flex items-center">
            <Brand />
            <div className="ml-auto flex">
              <button
                onClick={() => setSearchOpen((v) => !v)}
                className="grid h-9 w-9 place-items-center text-parchment/65 hover:text-parchment transition-colors"
                aria-label="Search chats"
              >
                <SearchIcon className="h-5 w-5" />
              </button>
              <button
                onClick={() => setLeftOpen(false)}
                className="grid h-9 w-9 place-items-center text-parchment/65 hover:text-parchment transition-colors"
                aria-label="Collapse sidebar"
              >
                <PanelIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
          {searchOpen && (
            <label className="relative mt-4 block">
              <SearchIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-parchment/45" />
              <input
                autoFocus
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search chats"
                className="w-full rounded-xl border border-white/10 bg-white/5 py-2.5 pl-9 pr-3 text-sm text-parchment outline-none"
              />
            </label>
          )}
          <button
            onClick={() => void addChat()}
            disabled={busy}
            className="group mt-6 flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium text-parchment/70 hover:bg-white/10 hover:text-parchment disabled:opacity-50 transition-colors"
          >
            <span className="text-lg">＋</span><span>New chat</span>
          </button>
          <div className="mt-6 flex min-h-0 flex-1 flex-col overflow-hidden">
            {pinned.length > 0 && (
              <>
                <p className="px-2 text-[11px] font-bold uppercase tracking-[.14em] text-parchment/35">Pinned chats</p>
                <div className="sidebar-scroll max-h-[35%] overflow-y-auto">
                  <ChatList chats={pinned} activeChatId={active} onSelect={setActive} onDelete={removeChat} onPin={pin} onRename={handleRenameChat} />
                </div>
              </>
            )}
            <p className="mt-2 px-2 text-[11px] font-bold uppercase tracking-[.14em] text-parchment/35">Recent chats</p>
            <div className="sidebar-scroll min-h-0 flex-1 overflow-y-auto">
              <ChatList chats={recent} activeChatId={active} onSelect={setActive} onDelete={removeChat} onPin={pin} onRename={handleRenameChat} />
            </div>
          </div>
          <div className="mt-3 flex items-center border-t border-white/10 pt-3">
            <button
              onClick={() => openUserProfile()}
              className="flex min-w-0 flex-1 items-center gap-3 rounded-xl px-2 py-2 hover:bg-white/[0.06] transition-colors"
              aria-label="Open profile settings"
            >
              <UserButton afterSignOutUrl="/" />
              <div className="min-w-0 text-left">
                <p className="truncate text-sm font-semibold text-parchment">{user?.fullName ?? user?.firstName ?? "Your profile"}</p>
                <p className="mt-0.5 truncate text-xs text-parchment/45">{user?.primaryEmailAddress?.emailAddress ?? ""}</p>
              </div>
            </button>
            <button
              onClick={() => signOut({ redirectUrl: "/" })}
              aria-label="Sign out"
              title="Sign out"
              className="ml-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-parchment/40 hover:bg-red-500/15 hover:text-red-400 transition-colors"
            >
              ↪
            </button>
          </div>
          {/* Face ID shortcut */}
          <button
            id="face-id-settings-btn"
            onClick={() => setFaceModalOpen(true)}
            className="mt-2 flex w-full items-center gap-2.5 rounded-xl border border-white/[0.07] bg-white/[0.04] px-3 py-2 text-parchment/55 transition hover:border-teal-500/30 hover:bg-teal-500/[0.08] hover:text-teal-300"
            aria-label="Manage Face ID"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4 shrink-0">
              <path d="M8 3H6a3 3 0 0 0-3 3v2M16 3h2a3 3 0 0 1 3 3v2M8 21H6a3 3 0 0 1-3-3v-2M16 21h2a3 3 0 0 0 3-3v-2M9 10h.01M15 10h.01M9.5 15c1.6 1.5 3.4 1.5 5 0" />
            </svg>
            <span className="text-xs font-medium">Face ID</span>
          </button>
        </div>

        {/* Collapsed Content: fixed 64px (w-16) inner width, icons centered */}
        <div
          className={`absolute inset-0 flex h-full w-16 flex-col items-center py-4 transition-all duration-300 ease-[cubic-bezier(0.2,0,0,1)] ${
            !leftOpen
              ? "opacity-100 translate-x-0 pointer-events-auto"
              : "opacity-0 -translate-x-4 pointer-events-none"
          }`}
        >
          {/* Logo area: QueryMind mark normally, changes to PanelIcon on hover, click expands */}
          <button
            type="button"
            onClick={() => {
              setLeftOpen(true);
              setCollapsedFlyout(null);
            }}
            title="Expand sidebar"
            aria-label="Expand sidebar"
            className="group relative grid h-10 w-10 place-items-center rounded-[14px] border border-teal-200/20 bg-[#0e5c59] text-parchment shadow-[0_10px_26px_rgba(12,184,169,.25)] transition-all duration-200 hover:bg-[#126b68] hover:border-teal-200/40 cursor-pointer overflow-hidden"
          >
            {/* Normal QueryMind logo: visible by default, smoothly fades out & scales down on hover */}
            <span className="absolute inset-0 grid place-items-center transition-all duration-200 opacity-100 scale-100 group-hover:opacity-0 group-hover:scale-75 pointer-events-none">
              <svg viewBox="0 0 36 36" className="h-7 w-7" fill="none" aria-hidden="true">
                <defs>
                  <linearGradient id="qm-mark-collapsed" x1="7" y1="5" x2="29" y2="31" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#e9fff9" />
                    <stop offset="1" stopColor="#7ce9dc" />
                  </linearGradient>
                </defs>
                <circle cx="16" cy="16" r="8.8" stroke="url(#qm-mark-collapsed)" strokeWidth="2.5" />
                <path d="m22.5 22.5 6.3 6.3" stroke="url(#qm-mark-collapsed)" strokeWidth="2.8" strokeLinecap="round" />
                <circle cx="12" cy="14" r="1.35" fill="#e9fff9" />
                <circle cx="18.4" cy="12" r="1.35" fill="#e9fff9" />
                <circle cx="17" cy="18.7" r="1.35" fill="#e9fff9" />
                <path d="m12.9 13.8 4.2-1.4m.4 1.2-.5 3.8m-3-2.3 2.1 2.5" stroke="#e9fff9" strokeWidth="1" opacity=".75" />
              </svg>
            </span>

            {/* Sidebar expand icon: hidden by default, smoothly fades in & scales up on hover */}
            <span className="absolute inset-0 grid place-items-center text-teal-100 transition-all duration-200 opacity-0 scale-75 group-hover:opacity-100 group-hover:scale-100 pointer-events-none">
              <PanelIcon className="h-5 w-5" />
            </span>
          </button>

          <div className="mt-4 w-8 border-t border-white/10" />

          {/* Action Icons */}
          <div className="mt-3 flex flex-col items-center gap-2">
            {/* New Chat */}
            <button
              onClick={() => void addChat()}
              disabled={busy}
              title="New chat"
              aria-label="New chat"
              className="grid h-10 w-10 place-items-center rounded-xl text-parchment/70 hover:bg-white/10 hover:text-parchment transition-all disabled:opacity-50"
            >
              <PlusIcon className="h-5 w-5" />
            </button>

            {/* Recent Uploads */}
            <button
              onClick={() => toggleFlyout("recent-uploads")}
              title="Recent uploads"
              aria-label="Recent uploads"
              className={`grid h-10 w-10 place-items-center rounded-xl transition-all ${
                collapsedFlyout === "recent-uploads"
                  ? "bg-white/15 text-parchment shadow-sm"
                  : "text-parchment/70 hover:bg-white/10 hover:text-parchment"
              }`}
            >
              <FileIcon className="h-5 w-5" />
            </button>

            {/* Search */}
            <button
              onClick={() => toggleFlyout("search")}
              title="Search chats"
              aria-label="Search chats"
              className={`grid h-10 w-10 place-items-center rounded-xl transition-all ${
                collapsedFlyout === "search"
                  ? "bg-white/15 text-parchment shadow-sm"
                  : "text-parchment/70 hover:bg-white/10 hover:text-parchment"
              }`}
            >
              <SearchIcon className="h-5 w-5" />
            </button>

            {/* Pinned */}
            <button
              onClick={() => toggleFlyout("pinned")}
              title="Pinned chats"
              aria-label="Pinned chats"
              className={`grid h-10 w-10 place-items-center rounded-xl transition-all ${
                collapsedFlyout === "pinned"
                  ? "bg-white/15 text-parchment shadow-sm"
                  : "text-parchment/70 hover:bg-white/10 hover:text-parchment"
              }`}
            >
              <PinIcon className="h-5 w-5" />
            </button>

            {/* Chats / History */}
            <button
              onClick={() => toggleFlyout("chats")}
              title="Chats / History"
              aria-label="Chats / History"
              className={`grid h-10 w-10 place-items-center rounded-xl transition-all ${
                collapsedFlyout === "chats"
                  ? "bg-white/15 text-parchment shadow-sm"
                  : "text-parchment/70 hover:bg-white/10 hover:text-parchment"
              }`}
            >
              <ChatIcon className="h-5 w-5" />
            </button>
          </div>

          {/* User Profile at bottom */}
          <div className="mt-auto flex flex-col items-center pb-1">
            <button
              onClick={() => openUserProfile()}
              title="User profile"
              aria-label="User profile"
              className="grid h-10 w-10 place-items-center rounded-xl hover:bg-white/10 transition-colors"
            >
              <UserButton afterSignOutUrl="/" />
            </button>
          </div>
        </div>
      </aside>

      {/* Collapsed flyout popover */}
      {!leftOpen && collapsedFlyout && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40 bg-black/20"
            onClick={() => setCollapsedFlyout(null)}
          />

          {/* Flyout panel */}
          <div className="fixed left-[72px] top-4 bottom-4 z-50 flex w-[300px] flex-col rounded-2xl border border-white/10 bg-[#12161b] p-4 text-parchment shadow-2xl backdrop-blur-xl animate-in fade-in slide-in-from-left-2 duration-200">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="text-xs font-bold uppercase tracking-[.12em] text-parchment/60">
                {collapsedFlyout === "search" && "Search chats"}
                {collapsedFlyout === "recent-uploads" && "Recent uploads"}
                {collapsedFlyout === "pinned" && "Pinned chats"}
                {collapsedFlyout === "chats" && "Chat history"}
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => {
                    setLeftOpen(true);
                    setCollapsedFlyout(null);
                  }}
                  title="Expand full sidebar"
                  aria-label="Expand full sidebar"
                  className="grid h-7 w-7 place-items-center rounded-lg text-parchment/55 transition-colors hover:bg-white/10 hover:text-parchment"
                >
                  <PanelIcon className="h-4 w-4" />
                </button>
                <button
                  onClick={() => setCollapsedFlyout(null)}
                  title="Close"
                  aria-label="Close"
                  className="grid h-7 w-7 place-items-center rounded-lg text-xs text-parchment/55 transition-colors hover:bg-white/10 hover:text-parchment"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Search flyout body */}
            {collapsedFlyout === "search" && (
              <div className="mt-3 flex min-h-0 flex-1 flex-col">
                <label className="relative block">
                  <SearchIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-parchment/45" />
                  <input
                    autoFocus
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Search chats..."
                    className="w-full rounded-xl border border-white/10 bg-white/5 py-2 pl-9 pr-3 text-sm text-parchment outline-none focus:border-teal-400/40"
                  />
                </label>
                <div className="sidebar-scroll mt-3 min-h-0 flex-1 overflow-y-auto">
                  <ChatList
                    chats={visible}
                    activeChatId={active}
                    onSelect={(id) => {
                      setActive(id);
                      setCollapsedFlyout(null);
                    }}
                    onDelete={removeChat}
                    onPin={pin}
                    onRename={handleRenameChat}
                  />
                  {visible.length === 0 && (
                    <p className="py-6 text-center text-xs text-parchment/40">No chats matching &quot;{search}&quot;</p>
                  )}
                </div>
              </div>
            )}

            {/* Recent Uploads flyout body */}
            {collapsedFlyout === "recent-uploads" && (
              <div className="mt-3 flex min-h-0 flex-1 flex-col">
                <div className="flex items-center justify-between pb-2">
                  <span className="text-[11px] text-parchment/50">
                    {storedDocs.length} {storedDocs.length === 1 ? "document" : "documents"}
                  </span>
                  {storedDocs.length > 0 && (
                    <button
                      onClick={() => {
                        if (selectedStoredDocs.length === storedDocs.length) {
                          setSelectedStoredDocs([]);
                        } else {
                          setSelectedStoredDocs(storedDocs.map((d) => d.filename));
                        }
                      }}
                      className="text-[11px] font-semibold text-teal-300 hover:underline"
                    >
                      {selectedStoredDocs.length === storedDocs.length ? "Deselect all" : "Select all"}
                    </button>
                  )}
                </div>

                <div className="sidebar-scroll min-h-0 flex-1 overflow-y-auto space-y-1 py-1">
                  {loadingStoredDocs ? (
                    <p className="py-6 text-center text-xs text-parchment/40">Loading documents...</p>
                  ) : storedDocs.length === 0 ? (
                    <p className="py-6 text-center text-xs text-parchment/40">No documents stored yet</p>
                  ) : (
                    storedDocs.map((doc) => {
                      const isChecked = selectedStoredDocs.includes(doc.filename);
                      return (
                        <label
                          key={doc.filename}
                          className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-xs cursor-pointer transition-colors ${
                            isChecked ? "bg-white/10 text-parchment" : "text-parchment/70 hover:bg-white/5"
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => {
                              setSelectedStoredDocs((prev) =>
                                prev.includes(doc.filename)
                                  ? prev.filter((f) => f !== doc.filename)
                                  : [...prev, doc.filename]
                              );
                            }}
                            className="rounded border-white/20 bg-transparent text-teal-400 focus:ring-0 focus:ring-offset-0"
                          />
                          <span className="truncate flex-1" title={doc.filename}>{doc.filename}</span>
                          {doc.size ? (
                            <span className="shrink-0 text-[10px] text-parchment/40">
                              {(doc.size / 1024).toFixed(0)} KB
                            </span>
                          ) : null}
                        </label>
                      );
                    })
                  )}
                </div>

                {storedDocs.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-white/10 flex items-center gap-2">
                    <button
                      onClick={handleSelectRecentFromFlyout}
                      disabled={selectedStoredDocs.length === 0}
                      className="flex-1 rounded-xl bg-teal-500 py-2 text-xs font-semibold text-slate-900 transition hover:bg-teal-400 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      Select {selectedStoredDocs.length > 0 ? `(${selectedStoredDocs.length})` : ""}
                    </button>
                    <button
                      onClick={() => void handleDeleteRecentFromFlyout()}
                      disabled={selectedStoredDocs.length === 0 || deletingStoredDocs}
                      className="flex-1 rounded-xl border border-red-500/30 bg-red-500/10 py-2 text-xs font-semibold text-red-300 transition hover:bg-red-500/20 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {deletingStoredDocs ? "Deleting..." : `Delete ${selectedStoredDocs.length > 0 ? `(${selectedStoredDocs.length})` : ""}`}
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Pinned flyout body */}
            {collapsedFlyout === "pinned" && (
              <div className="mt-3 flex min-h-0 flex-1 flex-col">
                <div className="sidebar-scroll min-h-0 flex-1 overflow-y-auto">
                  {pinned.length > 0 ? (
                    <ChatList
                      chats={pinned}
                      activeChatId={active}
                      onSelect={(id) => {
                        setActive(id);
                        setCollapsedFlyout(null);
                      }}
                      onDelete={removeChat}
                      onPin={pin}
                      onRename={handleRenameChat}
                    />
                  ) : (
                    <p className="py-6 text-center text-xs text-parchment/40">No pinned chats yet</p>
                  )}
                </div>
              </div>
            )}

            {/* Chats / History flyout body */}
            {collapsedFlyout === "chats" && (
              <div className="mt-3 flex min-h-0 flex-1 flex-col">
                <div className="sidebar-scroll min-h-0 flex-1 overflow-y-auto">
                  {recent.length > 0 ? (
                    <ChatList
                      chats={recent}
                      activeChatId={active}
                      onSelect={(id) => {
                        setActive(id);
                        setCollapsedFlyout(null);
                      }}
                      onDelete={removeChat}
                      onPin={pin}
                      onRename={handleRenameChat}
                    />
                  ) : (
                    <p className="py-6 text-center text-xs text-parchment/40">No chats yet</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </>
      )}

      <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
        {/* Right Sidebar toggle button: anchored at top-right of chat column in preview mode, or top-right corner */}
        <button
          onClick={handleToggleRightSidebar}
          className={`absolute top-4 z-30 grid h-9 w-9 place-items-center rounded-lg transition-all duration-300 ease-[cubic-bezier(0.2,0,0,1)] hover:bg-slate/5 ${
            rightOpen ? "text-slate" : "text-slate/40"
          } ${
            rightOpen && rightPanelView === "preview" && previewDoc
              ? "hidden md:grid md:right-[calc(50%+1rem)] xl:right-[calc(52%+1rem)] 2xl:right-[calc(55%+1rem)]"
              : "right-5"
          }`}
          aria-label="Toggle sources sidebar"
          title={rightOpen ? "Collapse sources sidebar" : "Expand sources sidebar"}
        >
          <PanelIcon className="h-5 w-5" />
        </button>

        <div className="flex min-h-0 flex-1 overflow-hidden">
          {/* ── Chat column ─────────────────────────────────────── */}
          {/*
           *  SCROLLBAR FIX:
           *  The section has NO horizontal padding now — padding was
           *  previously px-4/sm:px-8 which caused the scrollbar to float
           *  inward (inside the padded area) rather than flush with the
           *  right edge. Now the section fills edge-to-edge and the
           *  scrollbar docks against the section's own right border.
           *  Horizontal padding is applied on the inner content wrapper
           *  so messages still have comfortable margins.
           */}
          <section className="relative flex min-w-0 flex-1 flex-col py-6">

            {/* ── Message scroll container ─────────────────────── */}
            {/* This is the ONE scrollable element for chat messages. */}
            {/* scrollContainerRef is attached here, not to a sentinel. */}
            <div
              ref={scrollContainerRef}
              className="chat-scroll flex-1 overflow-y-auto [scrollbar-gutter:stable] w-full min-w-0"
            >
              {/* Inner wrapper carries the horizontal padding + max-width centering */}
              <div className="mx-auto flex w-full max-w-[800px] flex-col gap-5 px-4 pb-6 sm:px-8">
                {error && <div className="rounded-xl border border-red-300/30 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
                {messages.length > 0 ? (
                  <>
                    {messages.map((message) => (
                      <MessageBubble
                        key={message.id}
                        message={message}
                        onCitationClick={(cit) => {
                          handleOpenPreview({ filename: cit.label, snippet: cit.snippet }, cit);
                        }}
                      />
                    ))}
                    {busy && <ProcessingIndicator />}
                  </>
                ) : busy ? (
                  <ProcessingIndicator />
                ) : (
                  <div className="my-auto text-center">
                    <SparkleIcon className="mx-auto h-7 w-7 text-signal" />
                    <h1 className="mt-4 font-display text-3xl">What would you like to know?</h1>
                    <p className="mt-2 text-sm text-slate/50">Upload a document, then ask a question about it.</p>
                  </div>
                )}
                {/* Bottom sentinel — scroll target */}
                <div ref={bottomSentinelRef} aria-hidden="true" />
              </div>
            </div>

            {/* ── Scroll-to-bottom FAB ─────────────────────────── */}
            {/* Appears when user is not near the bottom. */}
            {/* Positioned absolutely inside the section, above the composer. */}
            {showScrollBtn && (
              <div
                className="pointer-events-none absolute bottom-[120px] left-0 right-0 z-20 flex justify-center"
                aria-hidden="true"
              >
                <button
                  onClick={() => scrollToBottom(true)}
                  className="pointer-events-auto scroll-to-bottom-btn"
                  aria-label="Scroll to latest message"
                  title="Scroll to bottom"
                >
                  {/* Down-chevron icon */}
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </button>
              </div>
            )}

            {/* ── Composer ─────────────────────────────────────── */}
            <div className="mx-auto w-full max-w-[800px] px-4 sm:px-8">
              <ChatInputBar
                onSend={(text) => void handleSend(text)}
                onFileSelected={(file) => void handleFile(file)}
                onRecallDoc={(filename) => void handleRecallDoc(filename)}
                onVoice={voice}
                listening={listening}
                attachedDocs={attachedDocs}
                onRemoveDoc={(id) => {
                  setAttachedDocs((prev) => prev.filter((d) => d.id !== id));
                  setError("");
                }}
                onDocsDeleted={(deletedNames) => {
                  setAttachedDocs((prev) => prev.filter((d) => !deletedNames.includes(d.name)));
                  if (previewDoc && deletedNames.includes(previewDoc.filename)) {
                    handleClosePreview();
                  }
                }}
              />
            </div>
          </section>

          {/* ── Right side: Document Preview OR Sources & Context ──────── */}
          {rightPanelView === "preview" && previewDoc ? (
            <aside
              className={`shrink-0 h-full border-l border-slate/10 transition-[width,opacity] duration-300 ease-[cubic-bezier(0.2,0,0,1)] overflow-hidden z-20 ${
                rightOpen
                  ? "w-full md:w-[50%] xl:w-[52%] 2xl:w-[55%] opacity-100 bg-[#0d1218]"
                  : "w-0 opacity-0 border-l-0 pointer-events-none"
              }`}
            >
              <div className="w-full h-full min-w-[360px]">
                <DocumentViewer
                  filename={previewDoc.filename}
                  initialSnippet={previewDoc.snippet}
                  onClose={handleClosePreview}
                />
              </div>
            </aside>
          ) : (
            <aside
              className={`shrink-0 h-full border-l border-slate/10 bg-white/40 transition-[width,opacity] duration-300 ease-[cubic-bezier(0.2,0,0,1)] overflow-hidden z-20 ${
                rightOpen ? "w-[320px] opacity-100" : "w-0 opacity-0 border-l-0 pointer-events-none"
              }`}
            >
              {/* Inner wrapper carries fixed 320px width so text and cards do not jump or wrap while collapsing */}
              <div className="right-sidebar-scroll w-[320px] h-full overflow-y-auto p-5 transition-opacity duration-200">
                <div className="flex items-center justify-between pr-10">
                  <p className="text-xs font-bold uppercase tracking-[.14em] text-slate/45">Sources &amp; context</p>
                  {sources.length > 0 && (
                    <span className="text-[11px] text-slate/40">{sources.length} document{sources.length > 1 ? "s" : ""}</span>
                  )}
                </div>
                {sources.length > 0 ? (
                  <div className="mt-4 flex flex-col gap-3">
                    {sources.map((src, i) => (
                      <div
                        key={i}
                        className="group rounded-2xl border border-signal/20 bg-white p-4 cursor-pointer hover:border-signal/50 hover:shadow-md transition-all"
                        onClick={() => {
                          handleOpenPreview(
                            { filename: src.label, snippet: src.snippet },
                            { id: `src-${i}`, label: src.label, snippet: src.snippet }
                          );
                        }}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <FileIcon className="h-4 w-4 text-signal shrink-0" />
                            <p className="text-sm font-semibold truncate">{src.label}</p>
                          </div>
                          <span className="text-[10px] font-semibold text-signal bg-signal/10 px-2 py-0.5 rounded opacity-80 group-hover:opacity-100 shrink-0">
                            Preview
                          </span>
                        </div>
                        <p className="mt-2 text-xs text-slate/55 line-clamp-3 leading-relaxed">{src.snippet}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="mt-5 rounded-2xl border border-dashed border-slate/20 p-5 text-center">
                    <FileIcon className="mx-auto h-6 w-6 text-slate/35" />
                    <p className="mt-3 text-sm">Upload a document and ask a question — sources will appear here.</p>
                  </div>
                )}
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
    {faceModalOpen && <FaceIdModal onClose={() => setFaceModalOpen(false)} />}
    {voiceOpen && (
      <VoiceRecorderModal
        onSend={(text) => { void handleSend(text); }}
        onClose={() => { setVoiceOpen(false); setListening(false); }}
      />
    )}
    </>
  );
}
