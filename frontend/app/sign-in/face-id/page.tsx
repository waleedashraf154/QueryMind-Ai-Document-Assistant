"use client";

import { useSignIn } from "@clerk/nextjs";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import Brand from "@/components/Brand";
import { ArrowIcon, FaceIcon } from "@/components/Icons";

export default function FaceIdSignInPage() {
  const { signIn, setActive, isLoaded } = useSignIn();
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "checking" | "error">("loading");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let mounted = true;
    navigator.mediaDevices?.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 640 } } })
      .then((stream) => {
        if (!mounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setStatus("ready");
      })
      .catch(() => {
        if (mounted) {
          setStatus("error");
          setErrorMessage("We couldn't access your camera. Check browser permissions and try again.");
        }
      });

    return () => {
      mounted = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  async function handleCapture() {
    if (!videoRef.current || !isLoaded || !signIn || !setActive) return;
    setStatus("checking");
    setErrorMessage("");

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      const res = await fetch("/api/face-sign-in", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageData: canvas.toDataURL("image/jpeg", 0.88) }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.error ?? "Face not recognized.");

      const result = await signIn.create({ strategy: "ticket", ticket: body.ticket });
      if (result.status !== "complete" || !result.createdSessionId) {
        throw new Error("Clerk sign-in did not complete.");
      }
      await setActive({ session: result.createdSessionId });
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      router.replace("/chat");
    } catch (error) {
      setStatus("ready");
      setErrorMessage(error instanceof Error ? error.message : "Something went wrong.");
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-ink px-5 py-10 text-parchment">
      <div className="absolute h-[480px] w-[480px] rounded-full bg-signal/20 blur-[120px]" />
      <main className="relative w-full max-w-lg rounded-[30px] border border-white/10 bg-[#1c1a29]/90 p-6 shadow-2xl sm:p-9">
        <div className="flex items-center justify-between"><Brand /><span className="rounded-full bg-teal-100/10 px-3 py-1 text-xs text-teal-100">Biometric access</span></div>
        <div className="mt-10 text-center">
          <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-signal/15 text-teal-200"><FaceIcon className="h-7 w-7" /></span>
          <h1 className="mt-5 font-display text-4xl font-semibold tracking-[-.04em]">Face ID sign in</h1>
          <p className="mx-auto mt-3 max-w-sm text-sm leading-6 text-parchment/65">Center your face in the frame. QueryMind will compare it with the Face ID you registered.</p>
        </div>
        <div className="relative mx-auto mt-8 aspect-square w-full max-w-[300px] overflow-hidden rounded-[48%] border border-teal-200/50 bg-[#0f1720] p-2 shadow-[0_0_0_12px_rgba(47,143,130,.08)]">
          <video ref={videoRef} autoPlay muted playsInline className="h-full w-full rounded-[46%] object-cover" />
          <div className="pointer-events-none absolute inset-4 rounded-[43%] border border-teal-200/50" />
        </div>
        {errorMessage && <p className="mt-5 text-center text-sm text-citation">{errorMessage}</p>}
        <button onClick={handleCapture} disabled={status !== "ready"} className="mt-8 flex w-full items-center justify-center gap-2 rounded-xl bg-signal px-6 py-3.5 font-semibold text-white transition hover:bg-teal-600 disabled:opacity-40">
          {status === "checking" ? "Verifying identity…" : "Verify and sign in"} <ArrowIcon className="h-4 w-4" />
        </button>
        <Link href="/sign-in" className="mt-5 block text-center text-sm text-parchment/55 transition hover:text-parchment">Use email and password instead</Link>
      </main>
    </div>
  );
}
