"""Reprocessa uma ou todas as normas com Vision (Opção C híbrida).

Uso:
    # Uma norma por ID
    python scripts/reprocess_norm.py d06b92ba-37c9-4c45-8bf1-e41633674559

    # Todas as normas
    python scripts/reprocess_norm.py --all

    # Dry-run (classifica páginas sem chamar Vision)
    python scripts/reprocess_norm.py --all --dry-run
"""
from __future__ import annotations
import asyncio
import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "ops_ai_grid.db"
PAGES_BASE = BACKEND / "knowledge" / "pages"


def db_connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_norm(conn, norm_id: str) -> sqlite3.Row | None:
    c = conn.cursor()
    c.execute("SELECT * FROM technical_norms WHERE id = ?", (norm_id,))
    return c.fetchone()


def get_all_norms(conn) -> list[sqlite3.Row]:
    c = conn.cursor()
    c.execute("SELECT * FROM technical_norms WHERE ativa = 1 ORDER BY codigo")
    return c.fetchall()


def update_norm_status(conn, norm_id: str, **kwargs):
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [norm_id]
    conn.execute(f"UPDATE technical_norms SET {fields} WHERE id = ?", values)
    conn.commit()


def delete_old_data(conn, norm_id: str):
    conn.execute("DELETE FROM norm_structures WHERE norm_id = ?", (norm_id,))
    conn.execute("DELETE FROM norm_materials WHERE norm_id = ?", (norm_id,))
    conn.commit()


def insert_structures(conn, norm_id: str, tenant_id: str, structures: list[dict]):
    now = datetime.now(timezone.utc).isoformat()
    struct_id_by_code: dict[str, str] = {}
    for s in structures:
        sid = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO norm_structures (
                id, created_at, updated_at, tenant_id, norm_id,
                codigo_estrutura, nome_completo, descricao_tecnica,
                caracteristicas_visuais, campos_proj, materiais,
                tipo_rede, tensao_nominal, como_identificar_na_foto,
                restricoes_uso, desenho_numero, pagina_referencia,
                imagem_desenho_path, fixacao, source_text_excerpt,
                extraction_confidence, requires_review
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sid, now, now, tenant_id, norm_id,
            str(s.get("codigo_estrutura", ""))[:80],
            str(s.get("nome_completo", ""))[:500],
            str(s.get("descricao_tecnica", "")),
            str(s.get("caracteristicas_visuais", "")),
            json.dumps({}), json.dumps([]),
            str(s.get("tipo_rede", ""))[:20],
            str(s.get("tensao_nominal", ""))[:40],
            str(s.get("como_identificar_na_foto", "")),
            str(s.get("restricoes_uso", "")),
            str(s.get("desenho_numero", ""))[:40],
            int(s.get("pagina_referencia", 0) or 0),
            str(s.get("imagem_desenho_path", "")),
            json.dumps(s.get("fixacao", {}) if isinstance(s.get("fixacao"), dict) else {}),
            str(s.get("source_text_excerpt", "")),
            float(s.get("extraction_confidence", 0.0) or 0.0),
            1 if s.get("requires_review") else 0,
        ))
        struct_id_by_code[str(s.get("codigo_estrutura", ""))] = sid
    conn.commit()
    return struct_id_by_code


def insert_materials(conn, norm_id: str, tenant_id: str,
                     materials_by_code: dict, struct_id_by_code: dict):
    now = datetime.now(timezone.utc).isoformat()
    for codigo, m in materials_by_code.items():
        ids = [struct_id_by_code[c] for c in m.get("structure_codes", [])
               if c in struct_id_by_code]
        conn.execute("""
            INSERT INTO norm_materials (
                id, created_at, updated_at, tenant_id, norm_id,
                codigo_material, codigo_item, descricao, tensao, used_in_structures
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            str(uuid.uuid4()), now, now, tenant_id, norm_id,
            str(codigo)[:80],
            str(m.get("codigo_item", ""))[:40],
            str(m.get("descricao", "")),
            str(m.get("tensao", ""))[:40],
            json.dumps(ids),
        ))
    conn.commit()


async def reprocess_one(norm_id: str, dry_run: bool = False):
    from app.services.norms_deep_extractor import deep_process_pdf, classify_pages

    conn = db_connect()
    norm = get_norm(conn, norm_id)
    if not norm:
        print(f"  ERRO: norma {norm_id} nao encontrada")
        conn.close()
        return False

    # Resolve path — relative to backend/
    pdf_path = BACKEND / norm["pdf_path"].replace("\\", "/")
    if not pdf_path.is_file():
        # try absolute as-is
        pdf_path2 = Path(norm["pdf_path"])
        if pdf_path2.is_file():
            pdf_path = pdf_path2
        else:
            print(f"  ERRO: PDF nao encontrado: {pdf_path}")
            conn.close()
            return False

    tenant_id = norm["tenant_id"]
    codigo = norm["codigo"]
    pages_dir = PAGES_BASE / norm_id
    pages_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Norma : {codigo}")
    print(f"  PDF   : {pdf_path.name}")
    print(f"  ID    : {norm_id}")
    print(f"{'='*60}")

    if dry_run:
        pages, doc = classify_pages(str(pdf_path))
        total = len(pages)
        drawing = [p for p in pages if p.is_drawing_page]
        print(f"  [DRY-RUN] {total} paginas, {len(drawing)} com desenhos")
        for p in drawing[:10]:
            print(f"    pag {p.page_num:3d}: imgs={p.image_count} draws={p.drawing_count} "
                  f"words={p.word_count} marker={p.has_marker}")
        if len(drawing) > 10:
            print(f"    ... e mais {len(drawing)-10}")
        doc.close()
        conn.close()
        return True

    update_norm_status(conn, norm_id,
        processing_status="classifying",
        processing_progress=0,
        processing_message="iniciando...",
        processing_started_at=datetime.now(timezone.utc).isoformat(),
        processing_finished_at=None,
    )

    last_print = [0.0]

    async def progress(step: str, current: int, total: int, message: str):
        pct = int(current / total * 100) if total > 0 else 0
        update_norm_status(conn, norm_id,
            processing_status=step,
            processing_progress=pct,
            processing_message=message[:500],
            pages_with_drawings=total if step == "extracting" else norm["pages_with_drawings"] or 0,
            pages_processed=current if step == "extracting" else 0,
        )
        now = time.monotonic()
        if now - last_print[0] > 0.5 or current == total:
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}% | {step} | {message[:50]}", end="", flush=True)
            last_print[0] = now

    t0 = time.monotonic()
    try:
        result = await deep_process_pdf(str(pdf_path), norm_id, pages_dir, progress)
    except Exception as e:
        print(f"\n  ERRO durante processamento: {e}")
        update_norm_status(conn, norm_id,
            processing_status="failed",
            processing_message=str(e)[:500],
            processing_finished_at=datetime.now(timezone.utc).isoformat(),
        )
        conn.close()
        return False

    elapsed = time.monotonic() - t0
    print()  # newline after progress bar

    # Persist
    delete_old_data(conn, norm_id)
    struct_map = insert_structures(conn, norm_id, tenant_id, result.structures)
    insert_materials(conn, norm_id, tenant_id, result.materials_by_code, struct_map)

    update_norm_status(conn, norm_id,
        processing_status="done",
        processing_progress=100,
        processing_message=(
            f"Concluido: {len(result.structures)} estruturas, "
            f"{len(result.materials_by_code)} materiais, "
            f"{result.pages_processed}/{result.pages_drawing} paginas Vision "
            f"de {result.pages_total} totais"
        ),
        processing_finished_at=datetime.now(timezone.utc).isoformat(),
        pages_total=result.pages_total,
        pages_with_drawings=result.pages_drawing,
        pages_processed=result.pages_processed,
    )

    print(f"  Resultado:")
    print(f"    Paginas totais      : {result.pages_total}")
    print(f"    Paginas com desenhos: {result.pages_drawing}")
    print(f"    Paginas Vision      : {result.pages_processed}")
    print(f"    Estruturas extraidas: {len(result.structures)}")
    print(f"    Materiais indexados : {len(result.materials_by_code)}")
    if result.page_errors:
        print(f"    Erros de pagina     : {len(result.page_errors)}")
        for pnum, err in result.page_errors[:3]:
            print(f"      pag {pnum}: {err[:80]}")
    print(f"    Tempo total         : {elapsed:.1f}s")

    conn.close()
    return True


async def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if not a.startswith("--")]

    if not args:
        print("Uso: python scripts/reprocess_norm.py <norm_id> | --all [--dry-run]")
        sys.exit(1)

    if args[0] == "--all" or args[0] == "all":
        conn = db_connect()
        norms = get_all_norms(conn)
        conn.close()
        print(f"Reprocessando {len(norms)} normas...")
        ok = err = 0
        for norm in norms:
            success = await reprocess_one(norm["id"], dry_run=dry_run)
            if success:
                ok += 1
            else:
                err += 1
        print(f"\n{'='*60}")
        print(f"CONCLUIDO: {ok} ok, {err} erros")
    else:
        norm_id = args[0]
        await reprocess_one(norm_id, dry_run=dry_run)


if __name__ == "__main__":
    asyncio.run(main())
