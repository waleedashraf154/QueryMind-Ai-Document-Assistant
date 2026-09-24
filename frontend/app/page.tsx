import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import Link from "next/link";

export default async function LandingPage() {
  const { userId } = await auth();

  // Face ID and email/password are alternatives, never two required steps.
  if (userId) {
    redirect("/chat");
  }

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center gap-8 overflow-hidden bg-ink px-6 text-parchment">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_20%,rgba(47,143,130,.24),transparent_32%)]" />
      <div className="relative text-center">
        <p className="mb-4 text-sm font-medium uppercase tracking-[.2em] text-teal-200">Document intelligence, refined</p>
        <h1 className="font-display text-5xl font-semibold tracking-[-.04em]">QueryMind</h1>
        <p className="mt-3 max-w-md text-parchment/70">
          Ask questions about your documents and get answers grounded in
          exactly what you uploaded.
        </p>
      </div>
      <div className="relative flex gap-4">
        <Link
          href="/sign-in"
          className="rounded-md border border-parchment/30 px-5 py-2.5 text-sm font-medium hover:bg-parchment/10"
        >
          Sign in
        </Link>
        <Link
          href="/sign-up"
          className="rounded-md bg-signal px-5 py-2.5 text-sm font-medium text-ink hover:opacity-90"
        >
          Get started
        </Link>
      </div>
    </main>
  );
}
