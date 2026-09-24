import { requireUserEmail } from "@/app/api/_lib/auth";
import { NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/**
 * GET /api/documents/status/[fileId]
 *
 * Proxies to backend GET /documents/status/{file_id}.
 *
 * Returns:
 *   {
 *     file_id: string;
 *     filename: string;
 *     status: "queued" | "processing" | "ready" | "failed" | "unknown";
 *     progress_message: string;
 *     error: string;
 *   }
 *
 * The frontend polls this endpoint every 2 seconds after calling
 * POST /api/upload/async, updating the file chip status badge accordingly.
 */
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ fileId: string }> }
) {
  try {
    // Auth check: only authenticated users can poll status
    await requireUserEmail();

    const { fileId } = await params;
    if (!fileId) {
      return NextResponse.json({ error: "fileId is required." }, { status: 400 });
    }

    const response = await fetch(
      `${API_URL}/documents/status/${encodeURIComponent(fileId)}`,
      { cache: "no-store" }
    );

    const data = await response.json().catch(() => ({}));
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    if (error instanceof Error && error.message === "UNAUTHORIZED") {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.json({ error: "Backend is unavailable." }, { status: 502 });
  }
}
