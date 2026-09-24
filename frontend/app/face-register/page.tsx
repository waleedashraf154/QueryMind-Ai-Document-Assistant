"use client";

import { useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import Brand from "@/components/Brand";
import { FaceIcon, ArrowIcon } from "@/components/Icons";
import { registerFace } from "@/lib/api";

export default function FaceRegisterPage() {
  const { user, isLoaded } = useUser();
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "checking" | "done" | "error">("loading");
  const [message, setMessage] = useState("");

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
          setMessage("Camera access was denied. Allow camera permission and try again.");
        }
      });
    return () => {
      mounted = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  async function capture() {
    if (!videoRef.current || !isLoaded || !user) return;
    setStatus("checking");
    setMessage("");

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      await registerFace(canvas.toDataURL("image/jpeg", 0.88));
      setStatus("done");
      setMessage("Face ID registered successfully. You can now use Face ID to sign in.");
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    } catch (error) {
      setStatus("ready");
      setMessage(error instanceof Error ? error.message : "Face registration failed.");
    }
  }

  return (
    <main className="relative flex min-h-screen items-center justify-center bg-ink px-5 py-10 text-parchment">
      <div className="relative w-full max-w-lg rounded-[30px] border border-white/10 bg-[#1c1a29]/95 p-6 shadow-2xl sm:p-9">
        <div className="flex items-center justify-between"><Brand /><span className="rounded-full bg-signal/15 px-3 py-1 text-xs text-teal-100">One-time setup</span></div>
        <div className="mt-9 text-center">
          <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-signal/15 text-teal-200"><FaceIcon className="h-7 w-7" /></span>
          <h1 className="mt-5 font-display text-4xl font-semibold tracking-[-.04em]">Register Face ID</h1>
          <p className="mx-auto mt-3 max-w-sm text-sm leading-6 text-parchment/65">
            {user?.firstName ? `Hi ${user.firstName}. ` : ""}Center only your face in the frame. This registers one biometric embedding for your QueryMind account.
          </p>
        </div>

        <div className="relative mx-auto mt-8 aspect-square w-full max-w-[300px] overflow-hidden rounded-[48%] border border-teal-200/50 bg-[#0f1720] p-2 shadow-[0_0_0_12px_rgba(47,143,130,.08)]">
          <video ref={videoRef} autoPlay muted playsInline className="h-full w-full rounded-[46%] object-cover" />
          <div className="pointer-events-none absolute inset-4 rounded-[43%] border border-teal-200/50" />
        </div>

        {message && <p className={`mt-5 text-center text-sm ${status === "done" ? "text-teal-200" : "text-citation"}`}>{message}</p>}

        {status !== "done" ? (
          <button onClick={capture} disabled={status !== "ready"} className="mt-8 flex w-full items-center justify-center gap-2 rounded-xl bg-signal px-6 py-3.5 font-semibold text-white transition hover:bg-teal-600 disabled:opacity-40">
            {status === "checking" ? "Registering…" : "Register Face ID"} <ArrowIcon className="h-4 w-4" />
          </button>
        ) : (
          <button onClick={() => router.push("/chat")} className="mt-8 w-full rounded-xl bg-signal px-6 py-3.5 font-semibold text-white">Continue to QueryMind</button>
        )}
        <button onClick={() => router.push("/chat")} className="mt-4 w-full text-sm text-parchment/50 hover:text-parchment">
          Skip for now
        </button>
      </div>
    </main>
  );
}
