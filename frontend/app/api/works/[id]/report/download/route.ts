import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { BACKEND_URL, API_PREFIX } from "@/lib/api";

export async function GET(
  req: NextRequest,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params;
  const fmt = req.nextUrl.searchParams.get("fmt") || "docx";

  const c = await cookies();
  const token = c.get("access_token")?.value;
  if (!token) return NextResponse.json({ error: "unauth" }, { status: 401 });

  const backendUrl = `${BACKEND_URL}${API_PREFIX}/works/${id}/report/download?fmt=${fmt}`;
  const r = await fetch(backendUrl, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });

  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: `Erro ${r.status}` }));
    return NextResponse.json(body, { status: r.status });
  }

  const blob = await r.arrayBuffer();
  const contentType =
    fmt === "pdf"
      ? "application/pdf"
      : "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

  const filename = `relatorio_${id.slice(0, 8)}.${fmt}`;

  return new NextResponse(blob, {
    status: 200,
    headers: {
      "Content-Type": contentType,
      "Content-Disposition": `attachment; filename="${filename}"`,
      "Content-Length": String(blob.byteLength),
    },
  });
}
