export type ChatRecord = {
  title: string;
  messages: { role: "user" | "assistant"; content: string }[];
};

async function parseResponse(res: Response) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error ?? data.detail ?? `Request failed: ${res.status}`);
  }
  return data;
}

export type Source = { label: string; snippet: string };

export type AskQuestionResponse = {
  answer: string;
  sources: Source[];
  chat_title?: string | null;
};

export async function askQuestion(chatId: string, question: string): Promise<AskQuestionResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, question }),
  });
  return parseResponse(res) as Promise<AskQuestionResponse>;
}

export async function uploadDocument(chatId: string, file: File) {
  const form = new FormData();
  form.append("chat_id", chatId);
  form.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  return parseResponse(res);
}

export type IngestionStatus = {
  file_id: string;
  filename: string;
  status: "queued" | "processing" | "ready" | "failed" | "unknown";
  progress_message: string;
  error: string;
};

/**
 * Upload a file and immediately get back a file_id.
 * OCR + embedding + indexing happen in the backend's background thread pool.
 * Poll getIngestionStatus(fileId) every ~2s to track progress.
 */
export async function uploadDocumentAsync(
  chatId: string,
  file: File
): Promise<{ file_id: string; filename: string; status: string }> {
  const form = new FormData();
  form.append("chat_id", chatId);
  form.append("file", file);
  const res = await fetch("/api/upload/async", { method: "POST", body: form });
  return parseResponse(res);
}

/**
 * Poll the ingestion status for a file uploaded via uploadDocumentAsync.
 * Returns immediately; caller loops with a delay between calls.
 */
export async function getIngestionStatus(fileId: string): Promise<IngestionStatus> {
  const res = await fetch(
    `/api/documents/status/${encodeURIComponent(fileId)}`,
    { cache: "no-store" }
  );
  return parseResponse(res) as Promise<IngestionStatus>;
}

export async function getChats() {
  const res = await fetch("/api/chats", { cache: "no-store" });
  return parseResponse(res) as Promise<Record<string, ChatRecord>>;
}

export async function createChat(chatId: string, title = "New Chat") {
  const res = await fetch("/api/chats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, title }),
  });
  return parseResponse(res);
}

export async function deleteChat(chatId: string) {
  const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}`, { method: "DELETE" });
  return parseResponse(res);
}

export async function renameChat(chatId: string, title: string) {
  const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return parseResponse(res);
}

export async function getChatSources(chatId: string): Promise<Source[]> {
  const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}/sources`, { cache: "no-store" });
  const data = await parseResponse(res);
  return (data.sources ?? []) as Source[];
}

export async function registerFace(imageData: string, overwrite = false) {
  const res = await fetch("/api/face-register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageData, overwrite }),
  });
  return parseResponse(res);
}

export async function getFaceStatus(): Promise<{ registered: boolean }> {
  const res = await fetch("/api/face-status", { cache: "no-store" });
  return parseResponse(res);
}

export async function deleteFace() {
  const res = await fetch("/api/face-delete", { method: "DELETE" });
  return parseResponse(res);
}

export type StoredDoc = {
  filename: string;
  size?: number;
  updated_at?: number;
};

export async function getStoredDocuments(): Promise<StoredDoc[]> {
  const res = await fetch("/api/documents", { cache: "no-store" });
  const data = await parseResponse(res);
  return (data.documents ?? []) as StoredDoc[];
}

export async function recallDocument(chatId: string, filename: string) {
  const res = await fetch("/api/documents/recall", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, filename }),
  });
  return parseResponse(res) as Promise<{ status: string; filename: string; char_count: number }>;
}

export async function deleteStoredDocuments(filenames: string[]): Promise<{ status: string; deleted: string[] }> {
  const res = await fetch("/api/documents", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filenames }),
  });
  return parseResponse(res);
}

export function getDocumentFileUrl(filename: string, preview = false): string {
  const base = `/api/documents/file?filename=${encodeURIComponent(filename)}`;
  return preview ? `${base}&preview=1` : base;
}

// Session in-memory cache for preview documents
const previewBufferCache = new Map<string, ArrayBuffer>();

export function clearPreviewBufferCache(filename?: string) {
  if (filename) {
    previewBufferCache.delete(filename);
  } else {
    previewBufferCache.clear();
  }
}

export async function fetchDocumentBytesForPreview(filename: string): Promise<ArrayBuffer> {
  if (previewBufferCache.has(filename)) {
    const cached = previewBufferCache.get(filename)!;
    console.log(`[PREVIEW] loaded from in-memory cache: '${filename}' (${cached.byteLength} bytes)`);
    return cached;
  }

  const res = await fetch("/api/documents/file", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Purpose": "preview",
    },
    body: JSON.stringify({ filename }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `Failed to fetch document (${res.status})`);
  }

  const contentType = res.headers.get("content-type") || "";
  let buffer: ArrayBuffer;

  if (contentType.includes("application/json")) {
    // Legacy / fallback json handling if ever returned
    const json = await res.json().catch(() => ({}));
    if (json.data && typeof json.data === "string") {
      const binary = atob(json.data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      buffer = bytes.buffer;
    } else {
      throw new Error("Invalid document preview response from server.");
    }
  } else {
    // Direct native C++ binary arrayBuffer streaming - zero base64 CPU overhead
    buffer = await res.arrayBuffer();
  }

  if (buffer.byteLength === 0) {
    throw new Error("Decoded document buffer is empty.");
  }

  // Cache in memory for instantaneous re-opens during session
  previewBufferCache.set(filename, buffer);
  return buffer;
}


export async function getDocumentPreview(filename: string): Promise<{ filename: string; text: string; size?: number; mime_type?: string }> {
  const res = await fetch(`/api/documents/preview?filename=${encodeURIComponent(filename)}`, { cache: "no-store" });
  return parseResponse(res);
}


