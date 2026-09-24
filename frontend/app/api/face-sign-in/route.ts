import { clerkClient } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function POST(req: Request) {
  try {
    const { imageData } = await req.json();
    if (!imageData) {
      return NextResponse.json({ error: "Face image is required." }, { status: 400 });
    }

    const matchRes = await fetch(`${API_URL}/face/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageData }),
      cache: "no-store",
    });
    const match = await matchRes.json().catch(() => ({}));

    if (!matchRes.ok || !match.email) {
      return NextResponse.json(
        { error: match.detail ?? "Face not recognized." },
        { status: matchRes.status || 401 },
      );
    }

    const client = await clerkClient();
    const users = await client.users.getUserList({ emailAddress: [match.email] });
    const user = users.data[0];
    if (!user) {
      return NextResponse.json({ error: "No Clerk account found for this face." }, { status: 404 });
    }

    const signInToken = await client.signInTokens.createSignInToken({
      userId: user.id,
      expiresInSeconds: 60,
    });

    return NextResponse.json({ ticket: signInToken.token });
  } catch (error) {
    console.error("Face sign-in error:", error);
    return NextResponse.json(
      { error: "Face sign-in is temporarily unavailable." },
      { status: 500 },
    );
  }
}
