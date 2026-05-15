import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { BACKEND_URL, API_PREFIX } from "@/lib/api";

export async function DELETE(
  _req: NextRequest,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params;
  const c = await cookies();
  const token = c.get("access_token")?.value;
  if (!token) return NextResponse.json({ error: "Unauthenticated" }, { status: 401 });

  const r = await fetch(`${BACKEND_URL}${API_PREFIX}/learning/${id}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (r.status === 204) return new NextResponse(null, { status: 204 });
  const data = await r.json().catch(() => ({}));
  return NextResponse.json(data, { status: r.status });
}
