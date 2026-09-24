import { SignIn } from "@clerk/nextjs";
import Link from "next/link";
import Brand from "@/components/Brand";
import { ArrowIcon, FaceIcon } from "@/components/Icons";

export default function SignInPage() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-ink px-5 py-10">
      {/* ambient glow */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_15%_20%,rgba(47,143,130,.3),transparent_28%),radial-gradient(circle_at_84%_80%,rgba(66,153,225,.18),transparent_28%)]" />

      <div className="relative w-full max-w-sm overflow-hidden rounded-[28px] border border-white/10 bg-parchment shadow-2xl">
        <section className="flex flex-col items-center px-8 py-10">

          {/* brand */}
          <div className="mb-8 self-start">
            <Brand />
          </div>

          {/* heading */}
          <div className="mb-6 w-full">
            <p className="text-xs font-semibold uppercase tracking-[.16em] text-signal">Choose your sign-in</p>
            <h1 className="mt-2 font-display text-3xl font-semibold tracking-[-.035em] text-ink">Welcome back</h1>
            <p className="mt-1.5 text-sm text-slate/60">Use Face ID or continue with your email.</p>
          </div>

          {/* Face ID button */}
          <Link
            href="/sign-in/face-id"
            className="group mb-6 flex w-full items-center justify-between rounded-2xl bg-ink p-4 text-parchment shadow-lg transition hover:-translate-y-0.5 hover:shadow-xl"
          >
            <span className="flex items-center gap-3">
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-signal text-white">
                <FaceIcon className="h-6 w-6" />
              </span>
              <span>
                <span className="block font-semibold">Sign in with Face ID</span>
                <span className="mt-0.5 block text-xs text-parchment/60">Fast, private biometric access</span>
              </span>
            </span>
            <ArrowIcon className="h-5 w-5 transition group-hover:translate-x-1" />
          </Link>

          {/* divider */}
          <div className="mb-5 flex w-full items-center gap-3 text-xs font-medium tracking-[.1em] text-slate/40 before:h-px before:flex-1 before:bg-slate/15 after:h-px after:flex-1 after:bg-slate/15">
            OR CONTINUE WITH EMAIL
          </div>

          {/* Clerk form — header hidden, form only */}
          <div className="w-full">
            <SignIn
              appearance={{
                variables: {
                  colorPrimary: "#2F8F82",
                  colorBackground: "#F3EEE2",
                  colorText: "#2A2833",
                  fontFamily: "var(--font-inter)",
                  borderRadius: "14px",
                },
                elements: {
                  rootBox: "w-full shadow-none",
                  cardBox: "shadow-none w-full",
                  card: "shadow-none border-0 bg-transparent p-0 gap-4",
                  header: "hidden",
                  headerTitle: "hidden",
                  headerSubtitle: "hidden",
                  main: "gap-3",
                  form: "gap-3",
                  socialButtonsBlockButton:
                    "border border-slate/20 bg-white text-slate font-medium rounded-xl h-11 shadow-none hover:bg-slate/5",
                  dividerLine: "bg-slate/15",
                  dividerText: "text-slate/40 text-xs tracking-widest",
                  formFieldLabel: "text-xs font-medium text-slate/70",
                  formFieldInput:
                    "rounded-xl border border-slate/20 bg-white text-slate placeholder:text-slate/35 h-11 focus:border-signal focus:ring-0 shadow-none",
                  formButtonPrimary:
                    "rounded-xl bg-signal h-11 text-sm font-semibold text-white hover:opacity-90 shadow-none",
                  footerActionLink: "text-signal font-medium",
                  footerActionText: "text-slate/55 text-sm",
                  identityPreviewEditButton: "text-signal",
                  formFieldRow: "gap-1",
                },
              }}
            />
          </div>
        </section>
      </div>
    </div>
  );
}
