import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = await auth();
  if (!isAuthenticated) redirect("/sign-in");
  return children;
}
