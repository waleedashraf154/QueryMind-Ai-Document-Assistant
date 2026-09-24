import { requireUserEmail } from "@/app/api/_lib/auth";
import { NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const { email } = await requireUserEmail();

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4000);

    try {
      const response = await fetch(
        `${API_URL}/face/status?email=${encodeURIComponent(email)}`,
        { cache: "no-store", signal: controller.signal },
      );
      clearTimeout(timer);
      const data = await response.json().catch(() => ({}));
      return NextResponse.json(data, { status: response.status });
    } catch {
      clearTimeout(timer);
      // Backend is down or timed out — treat as "not registered"
      return NextResponse.json({ registered: false }, { status: 200 });
    }
  } catch (error) {
    if (error instanceof Error && error.message === "UNAUTHORIZED") {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.json({ registered: false }, { status: 200 });
  }
}
