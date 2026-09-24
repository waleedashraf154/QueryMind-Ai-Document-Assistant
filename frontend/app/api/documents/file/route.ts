import { requireUserEmail } from "@/app/api/_lib/auth";
import { NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function GET(req: Request) {
  try {
    const { email } = await requireUserEmail();
    const { searchParams } = new URL(req.url);
    const filename = searchParams.get("filename");

    if (!filename) {
      return NextResponse.json({ error: "Filename is required." }, { status: 400 });
    }

    const reqHeaders: Record<string, string> = {};
    const clientRange = req.headers.get("range");
    if (clientRange) {
      reqHeaders["range"] = clientRange;
    }

    const res = await fetch(
      `${API_URL}/documents/${encodeURIComponent(email)}/file/${encodeURIComponent(filename)}`,
      {
        cache: "no-store",
        headers: reqHeaders,
      }
    );

    if (!res.ok && res.status !== 206) {
      const err = await res.json().catch(() => ({}));
      return NextResponse.json(err, { status: res.status });
    }

    const isPdf = filename.toLowerCase().endsWith(".pdf");
    const isDownload = searchParams.get("download") === "1" || searchParams.get("download") === "true";
    const isPreview = searchParams.get("preview") === "1" || req.headers.get("x-purpose") === "preview";

    let contentType = res.headers.get("content-type") || "application/octet-stream";
    if (isPdf) {
      // In preview mode, use application/octet-stream so external download managers (like IDM)
      // do not hijack the in-app document preview stream
      contentType = isPreview ? "application/octet-stream" : "application/pdf";
    }

    let disposition = isDownload
      ? `attachment; filename="${filename}"`
      : res.headers.get("content-disposition") || `inline; filename="${filename}"`;

    if (isPreview) {
      // For in-app preview, omit download filename disposition so browser extensions don't intercept
      disposition = "inline";
    } else if (isPdf && !isDownload) {
      disposition = `inline; filename="${filename}"`;
    }

    const buffer = await res.arrayBuffer();

    const responseHeaders: Record<string, string> = {
      "Content-Type": contentType,
      "Content-Disposition": disposition,
      "Cache-Control": "private, no-store, no-cache, must-revalidate",
      "Accept-Ranges": "bytes",
      "Content-Length": String(buffer.byteLength),
    };

    const contentRange = res.headers.get("content-range");
    if (contentRange) {
      responseHeaders["Content-Range"] = contentRange;
    }

    return new Response(new Uint8Array(buffer), {
      status: res.status,
      headers: responseHeaders,
    });
  } catch (error) {
    if (error instanceof Error && error.message === "UNAUTHORIZED") {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.json({ error: "Backend is unavailable." }, { status: 502 });
  }
}

export async function POST(req: Request) {
  try {
    const { email } = await requireUserEmail();
    const body = await req.json().catch(() => ({}));
    const filename = body.filename;

    if (!filename) {
      return NextResponse.json({ error: "Filename is required." }, { status: 400 });
    }

    const res = await fetch(
      `${API_URL}/documents/${encodeURIComponent(email)}/file/${encodeURIComponent(filename)}?preview=1`,
      {
        cache: "no-store",
        headers: {
          "x-purpose": "preview",
        },
      }
    );

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      return NextResponse.json(err, { status: res.status });
    }

    const arrayBuffer = await res.arrayBuffer();
    if (arrayBuffer.byteLength === 0) {
      return NextResponse.json({ error: "Document is empty." }, { status: 500 });
    }

    const mimeType = res.headers.get("content-type") || "application/octet-stream";

    return new Response(arrayBuffer, {
      status: 200,
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Length": String(arrayBuffer.byteLength),
        "Cache-Control": "private, max-age=3600",
        "X-Original-Mime": mimeType,
      },
    });
  } catch (error) {
    if (error instanceof Error && error.message === "UNAUTHORIZED") {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.json({ error: "Backend is unavailable." }, { status: 502 });
  }
}

