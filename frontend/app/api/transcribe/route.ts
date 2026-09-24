import { NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function POST(req: Request) {
  try {
    const incoming = await req.formData();
    const file = incoming.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "Audio file is required." }, { status: 400 });
    }

    const form = new FormData();
    form.append("file", file, file.name || "audio.webm");

    const response = await fetch(`${API_URL}/transcribe`, {
      method: "POST",
      body: form,
      cache: "no-store",
    });

    const data = await response.json().catch(() => ({}));
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("[api/transcribe] Error:", error);
    return NextResponse.json({ error: "Backend transcription unavailable." }, { status: 502 });
  }
}
