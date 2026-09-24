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

    const res = await fetch(
      `${API_URL}/documents/${encodeURIComponent(email)}/preview/${encodeURIComponent(filename)}`,
      { cache: "no-store" }
    );

    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, {
      status: res.status,
      headers: { "Cache-Control": "private, no-store, no-cache, must-revalidate" },
    });
  } catch (error) {
    if (error instanceof Error && error.message === "UNAUTHORIZED") {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.json({ error: "Backend is unavailable." }, { status: 502 });
  }
}
