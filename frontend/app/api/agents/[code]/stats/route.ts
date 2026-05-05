import { NextResponse } from "next/server";
import { backendAuthed } from "@/lib/api";

export async function GET(_: Request, ctx: { params: Promise<{ code: string }> }) {
  const { code } = await ctx.params;
  const r = await backendAuthed(`/agents/${code}/stats`);
  const data = await r.json().catch(() => ({}));
  return NextResponse.json(data, { status: r.status });
}
