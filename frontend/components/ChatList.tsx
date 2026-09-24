"use client";

import { useState, useRef, useEffect } from "react";

export type Chat = {
  id: string;
  title: string;
  pinned?: boolean;
};

export default function ChatList({
  chats,
  activeChatId,
  onSelect,
  onDelete,
  onPin,
  onRename,
}: {
  chats: Chat[];
  activeChatId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onPin: (id: string) => void;
  onRename?: (id: string, newTitle: string) => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const editContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingId]);

  function startRename(chat: Chat) {
    setEditingId(chat.id);
    setEditTitle(chat.title);
  }

  function handleConfirmRename(id: string) {
    const trimmed = editTitle.trim();
    if (trimmed) {
      const current = chats.find((c) => c.id === id);
      if (current && trimmed !== current.title) {
        onRename?.(id, trimmed);
      }
    }
    setEditingId(null);
  }

  function handleCancelRename() {
    setEditingId(null);
  }

  return (
    <div className="mt-3 space-y-1 pr-1 text-sm">
      {chats.map((chat) => {
        const isEditing = editingId === chat.id;

        return (
          <div
            key={chat.id}
            onClick={() => !isEditing && onSelect(chat.id)}
            className={`group flex items-center justify-between rounded-md px-3 py-2 transition-colors ${
              isEditing
                ? "bg-parchment/10 text-parchment cursor-default"
                : chat.id === activeChatId
                ? "bg-parchment/15 text-parchment cursor-pointer"
                : "text-parchment/60 hover:bg-parchment/10 hover:text-parchment cursor-pointer"
            }`}
          >
            {isEditing ? (
              <div
                ref={editContainerRef}
                className="flex w-full min-w-0 items-center gap-1.5"
                onClick={(e) => e.stopPropagation()}
              >
                <input
                  ref={inputRef}
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleConfirmRename(chat.id);
                    } else if (e.key === "Escape") {
                      e.preventDefault();
                      handleCancelRename();
                    }
                  }}
                  onBlur={(e) => {
                    if (editContainerRef.current?.contains(e.relatedTarget as Node)) {
                      return;
                    }
                    handleConfirmRename(chat.id);
                  }}
                  className="min-w-0 flex-1 rounded border border-teal-400/50 bg-black/40 px-2 py-0.5 text-xs text-parchment outline-none transition-all focus:border-teal-300 focus:ring-1 focus:ring-teal-300/30"
                />
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleConfirmRename(chat.id);
                  }}
                  title="Save title"
                  className="grid h-6 w-6 shrink-0 place-items-center rounded text-teal-300 transition-colors hover:bg-teal-500/20"
                >
                  <CheckIcon />
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCancelRename();
                  }}
                  title="Cancel"
                  className="grid h-6 w-6 shrink-0 place-items-center rounded text-parchment/40 transition-colors hover:bg-white/10 hover:text-parchment"
                >
                  <CloseIcon />
                </button>
              </div>
            ) : (
              <>
                <span className="truncate flex-1 min-w-0 pr-1">{chat.title}</span>
                <div className="flex shrink-0 items-center gap-0.5">
                  {onRename && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        startRename(chat);
                      }}
                      aria-label={`Rename ${chat.title}`}
                      title="Rename chat"
                      className="rounded p-1 text-parchment/40 opacity-0 transition-all duration-150 hover:text-teal-200 group-hover:opacity-100"
                    >
                      <EditIcon />
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onPin(chat.id);
                    }}
                    aria-label={`${chat.pinned ? "Unpin" : "Pin"} ${chat.title}`}
                    title={chat.pinned ? "Unpin chat" : "Pin chat"}
                    className={`rounded p-1 transition-all duration-150 hover:text-teal-200 ${
                      chat.pinned
                        ? "text-teal-200 opacity-100"
                        : "text-parchment/40 opacity-0 group-hover:opacity-100"
                    }`}
                  >
                    <PinIcon />
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(chat.id);
                    }}
                    aria-label={`Delete ${chat.title}`}
                    title="Delete chat"
                    className="rounded p-1 text-parchment/40 opacity-0 transition-all duration-150 hover:text-citation group-hover:opacity-100"
                  >
                    <TrashIcon />
                  </button>
                </div>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

function EditIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function PinIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m14 4 6 6-3 1-4 4-1 5-2-2-5-1 5-5 1-3 3-5Z" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0l-1 14a2 2 0 01-2 2H7a2 2 0 01-2-2L4 6h16z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
