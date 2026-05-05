import { redirect } from "next/navigation";
import { cookies } from "next/headers";

export default async function Home() {
  const c = await cookies();
  if (c.get("access_token")) redirect("/dashboard");
  redirect("/login");
}
