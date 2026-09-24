import { requireUserEmail } from "@/app/api/_lib/auth";
import { NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function POST(req: Request) {
  try {
    const { email } = await requireUserEmail();
    const { imageData, overwrite = false } = await req.json();

    const controller = new AbortController();
    // Face registration runs ML inference — allow up to 30s
    const timer = setTimeout(() => controller.abort(), 30_000);

    try {
      const response = await fetch(`${API_URL}/face/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, imageData, overwrite }),
        cache: "no-store",
        signal: controller.signal,
      });
      clearTimeout(timer);
      const data = await response.json().catch(() => ({}));
      return NextResponse.json(data, { status: response.status });
    } catch {
      clearTimeout(timer);
      return NextResponse.json(
        { error: "Face registration timed out. The backend may be starting up — please try again." },
        { status: 504 },
      );
    }
  } catch (error) {
    if (error instanceof Error && error.message === "UNAUTHORIZED") {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.json({ error: "Face registration service is unavailable." }, { status: 502 });
  }
}
