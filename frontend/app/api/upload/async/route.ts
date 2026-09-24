import { requireUserEmail } from "@/app/api/_lib/auth";
import { NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/**
 * POST /api/upload/async
 *
 * Proxies to the backend POST /upload/async endpoint, which:
 *   1. Saves the file bytes to disk immediately
 *   2. Fires off Supabase storage upload in a background daemon thread
 *   3. Queues OCR + chunking + embedding + Chroma indexing in a bounded
 *      ThreadPoolExecutor (max 2 workers)
 *   4. Returns { file_id, filename, status: "queued" } within ~200ms
 *
 * The client then polls GET /api/documents/status/[fileId] every 2 seconds
 * to track progress and update the per-chip status badge.
 */
export async function POST(req: Request) {
  try {
    const { email } = await requireUserEmail();
    const incoming = await req.formData();
    const chatId = String(incoming.get("chat_id") ?? "");
    const file = incoming.get("file");

    if (!chatId || !(file instanceof File)) {
      return NextResponse.json(
        { error: "chat_id and file are required." },
        { status: 400 }
      );
    }

    const form = new FormData();
    form.append("chat_id", chatId);
    form.append("user_email", email);
    form.append("file", file, file.name);

    const response = await fetch(`${API_URL}/upload/async`, {
      method: "POST",
      body: form,
      cache: "no-store",
    });

    const data = await response.json().catch(() => ({}));
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    if (error instanceof Error && error.message === "UNAUTHORIZED") {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.json({ error: "Backend is unavailable." }, { status: 502 });
  }
}
