import { NextRequest, NextResponse } from "next/server";
import { backendAuthed } from "@/lib/api";

export async function POST(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const body = await req.json().catch(() => ({}));
  const r = await backendAuthed(`/knowledge/${id}/deactivate`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  return NextResponse.json(data, { status: r.status });
}
