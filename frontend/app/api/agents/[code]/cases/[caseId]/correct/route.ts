import { NextRequest, NextResponse } from "next/server";
import { backendAuthed } from "@/lib/api";

export async function POST(req: NextRequest, ctx: { params: Promise<{ code: string; caseId: string }> }) {
  const { code, caseId } = await ctx.params;
  const body = await req.json();
  const r = await backendAuthed(`/agents/${code}/cases/${caseId}/correct`, {
    method: "POST",
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  return NextResponse.json(data, { status: r.status });
}
