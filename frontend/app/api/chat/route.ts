import { requireUserEmail } from "@/app/api/_lib/auth";
import { NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function POST(req: Request) {
  try {
    const { email } = await requireUserEmail();
    const body = await req.json();
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: body.chat_id,
        user_email: email,
        question: body.question,
        selected_source: body.selected_source,
      }),
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
