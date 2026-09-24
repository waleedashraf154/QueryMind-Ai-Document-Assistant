import { auth, clerkClient } from "@clerk/nextjs/server";

export async function requireUserEmail() {
  const { userId } = await auth();
  if (!userId) throw new Error("UNAUTHORIZED");

  const client = await clerkClient();
  const user = await client.users.getUser(userId);
  const email = user.primaryEmailAddress?.emailAddress;
  if (!email) throw new Error("Your Clerk account has no primary email address.");
  return { userId, email };
}

