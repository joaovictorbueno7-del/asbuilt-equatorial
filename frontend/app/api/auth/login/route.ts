import { NextRequest, NextResponse } from "next/server";
import { backendFetch } from "@/lib/api";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const r = await backendFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));

  if (!r.ok) {
    return NextResponse.json(
      { error: typeof data.detail === "string" ? data.detail : "Erro ao autenticar" },
      { status: r.status },
    );
  }

  if (data.mfa_required) {
    return NextResponse.json({ mfa_required: true });
  }

  const res = NextResponse.json({ ok: true });
  const secure = process.env.NODE_ENV === "production";
  res.cookies.set("access_token", data.access_token, {
    httpOnly: true,
    sameSite: "lax",
    secure,
    path: "/",
    maxAge: 60 * (data.expires_in_minutes ?? 15),
  });
  res.cookies.set("refresh_token", data.refresh_token, {
    httpOnly: true,
    sameSite: "lax",
    secure,
    path: "/",
    maxAge: 60 * 60 * 24 * 7,
  });
  return res;
}
