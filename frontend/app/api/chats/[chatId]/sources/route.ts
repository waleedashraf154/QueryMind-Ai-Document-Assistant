import { requireUserEmail } from "@/app/api/_lib/auth";
import { NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ chatId: string }> },
) {
  try {
    const { email } = await requireUserEmail();
    const { chatId } = await params;
    const response = await fetch(
      `${API_URL}/chats/${encodeURIComponent(chatId)}/sources?user_email=${encodeURIComponent(email)}`,
      { cache: "no-store" }
    );
    const data = await response.json().catch(() => ({ sources: [] }));
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    if (error instanceof Error && error.message === "UNAUTHORIZED") {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.json({ sources: [] }, { status: 200 });
  }
}
