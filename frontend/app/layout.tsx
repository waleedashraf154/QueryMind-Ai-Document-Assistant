import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import { Fraunces, Inter } from "next/font/google";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  weight: ["400", "500", "600"],
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "QueryMind",
  description: "AI document assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    // ClerkProvider makes sign-in state available to every component
    // below it via hooks like useUser() and useAuth() — this is the
    // single line that "turns on" Clerk for the whole app.
    <ClerkProvider signInUrl="/sign-in" signUpUrl="/sign-up">
      <html lang="en">
        <body className={`${fraunces.variable} ${inter.variable} font-sans`}>
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}
