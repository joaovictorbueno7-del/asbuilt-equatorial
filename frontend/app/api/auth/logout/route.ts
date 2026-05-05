import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { backendFetch } from "@/lib/api";

export async function POST() {
  const c = await cookies();
  const access = c.get("access_token")?.value;
  const refresh = c.get("refresh_token")?.value;

  if (access && refresh) {
    await backendFetch("/auth/logout", {
      method: "POST",
      headers: { Authorization: `Bearer ${access}` },
      body: JSON.stringify({ refresh_token: refresh }),
    }).catch(() => {});
  }

  const res = NextResponse.json({ ok: true });
  res.cookies.delete("access_token");
  res.cookies.delete("refresh_token");
  return res;
}
