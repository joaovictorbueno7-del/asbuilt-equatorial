import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { backendAuthed, BACKEND_URL, API_PREFIX } from "@/lib/api";

export async function GET(req: NextRequest) {
  const qs = req.nextUrl.search;
  const r = await backendAuthed(`/knowledge${qs}`);
  const data = await r.json().catch(() => ({}));
  return NextResponse.json(data, { status: r.status });
}

export async function POST(req: NextRequest) {
  const c = await cookies();
  const token = c.get("access_token")?.value;
  if (!token) return NextResponse.json({ error: "Unauthenticated" }, { status: 401 });
  const fd = await req.formData();
  const r = await fetch(`${BACKEND_URL}${API_PREFIX}/knowledge`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: fd as unknown as BodyInit,
  });
  const data = await r.json().catch(() => ({}));
  return NextResponse.json(data, { status: r.status });
}
