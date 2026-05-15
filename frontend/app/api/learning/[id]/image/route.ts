import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { BACKEND_URL, API_PREFIX } from "@/lib/api";

export async function GET(
  _req: NextRequest,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params;
  const c = await cookies();
  const token = c.get("access_token")?.value;
  if (!token) return NextResponse.json({ error: "Unauthenticated" }, { status: 401 });

  const r = await fetch(`${BACKEND_URL}${API_PREFIX}/learning/${id}/image`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });
  if (!r.ok) return NextResponse.json({ error: "not found" }, { status: r.status });
  const blob = await r.arrayBuffer();
  return new NextResponse(blob, {
    status: 200,
    headers: {
      "Content-Type": r.headers.get("content-type") || "image/jpeg",
      "Cache-Control": "private, max-age=3600",
    },
  });
}
