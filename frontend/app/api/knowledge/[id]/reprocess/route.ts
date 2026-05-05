import { NextResponse } from "next/server";
import { backendAuthed } from "@/lib/api";

export async function POST(_: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const r = await backendAuthed(`/knowledge/${id}/reprocess`, { method: "POST" });
  const data = await r.json().catch(() => ({}));
  return NextResponse.json(data, { status: r.status });
}
