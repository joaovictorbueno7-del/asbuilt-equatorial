import { NextResponse } from "next/server";
import { backendAuthed } from "@/lib/api";

export async function GET() {
  const r = await backendAuthed("/notifications?limit=20");
  const data = await r.json().catch(() => ({}));
  return NextResponse.json(data, { status: r.status });
}
