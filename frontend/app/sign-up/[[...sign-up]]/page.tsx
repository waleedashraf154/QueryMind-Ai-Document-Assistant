import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-ink">
      <SignUp
        forceRedirectUrl="/face-register"
        fallbackRedirectUrl="/face-register"
        appearance={{
          variables: {
            colorPrimary: "#2F8F82",
            colorBackground: "#F3EEE2",
            colorText: "#2A2833",
            fontFamily: "var(--font-inter)",
          },
        }}
      />
    </div>
  );
}
