"""Indexação dirigida NT.00022 — vai direto nas 94 estruturas conhecidas.

Usa as imagens já renderizadas em backend/knowledge/pages/{norm_id}/
e envia cada uma para Claude Vision com prompt direcionado.

Uso:
    python scripts/index_nt00022_targeted.py
    python scripts/index_nt00022_targeted.py --dry-run
"""
from __future__ import annotations
import asyncio
import base64
import json
import re
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

NORM_ID = "d06b92ba-37c9-4c45-8bf1-e41633674559"
NORM_CODE = "NT.00022.EQTL-04"
PAGES_DIR = BACKEND / "knowledge" / "pages" / NORM_ID
DB_PATH = ROOT / "ops_ai_grid.db"

# Mapa: código → número da página (ou lista de páginas para estruturas multi-página)
PAGE_MAP: dict[str, int | list[int]] = {
    # BT
    "S1I": 93, "S1I-45": 94, "S3I": 95, "S1I-S3I-TAN": 96,
    "S1I-S3I-OP": 98, "S4I-90": 100, "S4I-CF": 101, "S4I": 102,
    "SI-M": 104, "SEC-AEREO": 105, "FLY-TAP": 106,
    # MT-MONO
    "U1": 112, "U2": 113, "U3": 114, "U4": 115, "U3.2": 116,
    "U3-U3": 117, "UP1": 118, "UP3": 119, "UP4": 120,
    "UT1-UP3": 121, "UT1": 122, "UP1-UP3": 123,
    # MT-BI
    "BP1-A": 124, "BP1": 125, "BP3": 127, "2BP3": 129,
    "BP4": 131, "BP1-BP3": 133, "BP1A-PR": 135, "N4T-B": 142,
    # MT-BI-BECO
    "B1B": 137, "B2B": 138, "B3B": 139, "B4B": 140, "HTEB": 141,
    # MT-TRI
    "N1": 143, "N2": 144, "N3": 145, "N4": 146, "B1": 148,
    "B2": 149, "B3": 150, "B3-B3": 151, "B4": 153, "N1-N1": 154,
    "N1-DN3": 155, "N3-N3": 157, "2N1": 159, "N1-N1-DN3": 160,
    "N3-N3-N1": 161, "T1": 162, "T2": 163, "T3": 164, "T4": 166,
    "TE": 168, "HT": 170, "HTE": 172,
    # MT-TRI-PILAR
    "P1A": 173, "P1": 174, "PT1": 175, "PTA1": 176, "P3": 177,
    "P3.P3": 178, "P4": 179, "P1.P3": 180, "PT1.P3": 182,
    "P1.P1": 184, "P1A-PR": 185, "PT1.PR": 187, "PTA1.PR": 189,
    # EQUIP
    "CHAVE-FACA-UNI": 195, "CF-N1-U3": 196, "CF-U1-U3": 197,
    "CF-U4": 198, "CF-N4": 199, "PR-N4": 201, "CF-RAMAL": 202,
    "CF-ALINHA": 204,
    # TRAFO
    "TR-MONO-FIM": 205, "TR-MONO-TAN": 207, "B1-BS": 209, "N1-NS": 211,
    # ESPECIAL
    "UP4-CR": 215, "N4B-NS-CR": 216, "N4-NS-CR": 218, "N4B-NSCF": 219,
    "N4B-SU": 221, "N1B-N3B-CF": 222, "FF-N1B-NSCF-TM": 224,
    "FF-N3B-NSCF-TM": [226, 228],
    # MUFLA
    "MUFLA-CF": 230, "MUFLA-CFA": 232,
}

GRUPOS: dict[str, str] = {}  # código → grupo
_GRUPO_MAP = {
    "BT": ["S1I", "S1I-45", "S3I", "S1I-S3I-TAN", "S1I-S3I-OP", "S4I-90",
           "S4I-CF", "S4I", "SI-M", "SEC-AEREO", "FLY-TAP"],
    "MT-MONO": ["U1", "U2", "U3", "U4", "U3.2", "U3-U3", "UP1", "UP3",
                "UP4", "UT1-UP3", "UT1", "UP1-UP3"],
    "MT-BI": ["BP1-A", "BP1", "BP3", "2BP3", "BP4", "BP1-BP3", "BP1A-PR", "N4T-B"],
    "MT-BI-BECO": ["B1B", "B2B", "B3B", "B4B", "HTEB"],
    "MT-TRI": ["N1", "N2", "N3", "N4", "B1", "B2", "B3", "B3-B3", "B4",
               "N1-N1", "N1-DN3", "N3-N3", "2N1", "N1-N1-DN3", "N3-N3-N1",
               "T1", "T2", "T3", "T4", "TE", "HT", "HTE"],
    "MT-TRI-PILAR": ["P1A", "P1", "PT1", "PTA1", "P3", "P3.P3", "P4",
                     "P1.P3", "PT1.P3", "P1.P1", "P1A-PR", "PT1.PR", "PTA1.PR"],
    "EQUIP": ["CHAVE-FACA-UNI", "CF-N1-U3", "CF-U1-U3", "CF-U4", "CF-N4",
              "PR-N4", "CF-RAMAL", "CF-ALINHA"],
    "TRAFO": ["TR-MONO-FIM", "TR-MONO-TAN", "B1-BS", "N1-NS"],
    "ESPECIAL": ["UP4-CR", "N4B-NS-CR", "N4-NS-CR", "N4B-NSCF", "N4B-SU",
                 "N1B-N3B-CF", "FF-N1B-NSCF-TM", "FF-N3B-NSCF-TM"],
    "MUFLA": ["MUFLA-CF", "MUFLA-CFA"],
}
_TIPO_REDE_MAP = {
    "BT": "BT",
    "MT-MONO": "MT", "MT-BI": "MT", "MT-BI-BECO": "MT", "MT-TRI": "MT", "MT-TRI-PILAR": "MT",
    "EQUIP": "MT", "TRAFO": "MT", "ESPECIAL": "MT", "MUFLA": "MT",
}
for grupo, codes in _GRUPO_MAP.items():
    for c in codes:
        GRUPOS[c] = grupo

TARGETED_PROMPT = """Você é especialista em normas técnicas de distribuição elétrica da Equatorial (NT.00022.EQTL-04).

Esta página mostra a estrutura elétrica: {codigo} (grupo: {grupo} / rede: {tipo_rede})

Extraia EXATAMENTE um JSON com este schema (NADA antes ou depois do JSON):
{{
  "nome_completo": "nome completo conforme aparece no desenho",
  "descricao_tecnica": "o que é e como funciona (máx 400 chars)",
  "tensao_nominal": "tensão do sistema (ex: '13.8kV', '220/127V', '34.5kV') ou ''",
  "como_identificar_na_foto": "como reconhecer esta estrutura numa foto de campo — detalhes visuais específicos: formato da cruzeta, posição dos isoladores, tipo de cabo, etc.",
  "caracteristicas_visuais": "características visuais: dimensões aproximadas, materiais visíveis, configuração espacial dos componentes",
  "materiais": [
    {{"codigo_item": "codigo do catálogo ou N/D", "descricao": "descrição do item", "quantidade": "qtd+unidade", "tensao": "tensão ou ''"}}
  ],
  "fixacao": {{"tipo": "topo|lateral|console", "altura_m": 0}},
  "restricoes_uso": "quando NÃO usar (deixe '' se não houver)",
  "desenho_numero": "número do desenho se visível ou ''"
}}

Regras:
- materiais: se houver lista de materiais/BOM na página, extraia TODOS os itens
- Se a página tiver múltiplas estruturas, foque em: {codigo}
- NUNCA invente dados não visíveis na página
- O JSON deve ter exatamente esses campos
- Retorne SOMENTE o JSON"""

MAX_PARALLEL = 2  # conservador para respeitar rate limit
SEMAPHORE = asyncio.Semaphore(MAX_PARALLEL)


def db_connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_page_image(page_num: int) -> bytes | None:
    path = PAGES_DIR / f"page_{page_num:04d}.jpg"
    if not path.is_file():
        print(f"  [AVISO] Página {page_num} não encontrada: {path.name}")
        return None
    return path.read_bytes()


def _extract_json(text: str) -> dict | None:
    # Remove markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def vision_extract_structure(
    codigo: str,
    page_num: int | list[int],
    dry_run: bool = False,
) -> dict | None:
    """Envia página(s) para Claude Vision e extrai dados da estrutura."""
    from anthropic import AsyncAnthropic
    from agents.kmz_analyzer.vision import _resolve_api_key, _normalize_image

    grupo = GRUPOS.get(codigo, "DESCONHECIDO")
    tipo_rede = _TIPO_REDE_MAP.get(grupo, "MT")

    # Carrega imagens
    pages = [page_num] if isinstance(page_num, int) else page_num
    images_data = []
    for pnum in pages:
        data = load_page_image(pnum)
        if data:
            images_data.append((pnum, data))

    if not images_data:
        print(f"  [ERRO] Nenhuma imagem disponível para {codigo} (páginas: {pages})")
        return None

    if dry_run:
        print(f"  [DRY-RUN] {codigo} | grupo={grupo} | página(s)={pages} | imagens={len(images_data)}")
        return {"dry_run": True, "codigo": codigo, "paginas": pages}

    key = _resolve_api_key()
    if not key:
        print("  [ERRO] ANTHROPIC_API_KEY não configurada")
        return None

    client = AsyncAnthropic(api_key=key)
    prompt = TARGETED_PROMPT.format(
        codigo=codigo, grupo=grupo, tipo_rede=tipo_rede
    )

    # Constrói conteúdo: imagens + prompt
    content = []
    for pnum, raw in images_data:
        img_bytes, media_type = _normalize_image(raw)
        b64 = base64.standard_b64encode(img_bytes).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64}
        })
        content.append({"type": "text", "text": f"(página {pnum} da norma NT.00022 — estrutura {codigo})"})

    content.append({"type": "text", "text": prompt})

    async with SEMAPHORE:
        try:
            msg = await client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2000,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                print(f"  [429] {codigo} — aguardando 60s...")
                await asyncio.sleep(60)
                try:
                    msg = await client.messages.create(
                        model="claude-sonnet-4-5",
                        max_tokens=2000,
                        messages=[{"role": "user", "content": content}],
                    )
                except Exception as e2:
                    print(f"  [ERRO] {codigo} falhou novamente: {e2}")
                    return None
            else:
                print(f"  [ERRO] {codigo} API error: {e}")
                return None

    raw_text = "".join(getattr(b, "text", "") for b in msg.content)
    usage = getattr(msg, "usage", None)
    tok = f"in={usage.input_tokens} out={usage.output_tokens}" if usage else ""
    parsed = _extract_json(raw_text)

    if parsed is None:
        print(f"  [WARN] {codigo} sem JSON ({tok}): {raw_text[:80]}")
        return None

    # Adiciona metadados
    parsed["codigo_estrutura"] = codigo
    parsed["tipo_rede"] = tipo_rede
    parsed["grupo"] = grupo
    parsed["pagina_referencia"] = pages[0]
    parsed["imagem_desenho_path"] = str(PAGES_DIR / f"page_{pages[0]:04d}.jpg")
    print(f"  [OK] {codigo} | {tok} | {len(parsed.get('materiais', []))} materiais")
    return parsed


def delete_old(conn, norm_id: str):
    conn.execute("DELETE FROM norm_structures WHERE norm_id = ?", (norm_id,))
    conn.execute("DELETE FROM norm_materials WHERE norm_id = ?", (norm_id,))
    conn.commit()


def insert_structure(conn, norm_id: str, tenant_id: str, s: dict) -> str:
    now = datetime.now(timezone.utc).isoformat()
    sid = str(uuid.uuid4())
    materiais_list = s.get("materiais", [])
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
        json.dumps({}), json.dumps(materiais_list),
        str(s.get("tipo_rede", ""))[:20],
        str(s.get("tensao_nominal", ""))[:40],
        str(s.get("como_identificar_na_foto", "")),
        str(s.get("restricoes_uso", "")),
        str(s.get("desenho_numero", ""))[:40],
        int(s.get("pagina_referencia", 0) or 0),
        str(s.get("imagem_desenho_path", "")),
        json.dumps(s.get("fixacao", {}) if isinstance(s.get("fixacao"), dict) else {}),
        "",  # source_text_excerpt
        0.85,  # extraction_confidence (Vision dirigida = alta)
        0,     # requires_review
    ))
    conn.commit()
    return sid


def insert_materials(conn, norm_id: str, tenant_id: str, struct_id: str, codigo: str, materiais: list):
    now = datetime.now(timezone.utc).isoformat()
    for m in materiais:
        if not isinstance(m, dict):
            continue
        codigo_item = str(m.get("codigo_item", "N/D"))[:40]
        descricao = str(m.get("descricao", ""))
        quantidade = str(m.get("quantidade", ""))
        tensao = str(m.get("tensao", ""))[:40]
        # codigo_material = código estrutura + sufixo do item
        codigo_material = f"{codigo}:{codigo_item}"[:80]
        conn.execute("""
            INSERT INTO norm_materials (
                id, created_at, updated_at, tenant_id, norm_id,
                codigo_material, codigo_item, descricao, tensao, used_in_structures
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            str(uuid.uuid4()), now, now, tenant_id, norm_id,
            codigo_material, codigo_item, descricao, tensao,
            json.dumps([struct_id]),
        ))
    conn.commit()


async def run(dry_run: bool = False):
    conn = db_connect()
    c = conn.cursor()
    c.execute("SELECT * FROM technical_norms WHERE id = ?", (NORM_ID,))
    norm = c.fetchone()
    if not norm:
        print(f"ERRO: norma {NORM_ID} não encontrada no banco")
        conn.close()
        return

    tenant_id = norm["tenant_id"]
    total = len(PAGE_MAP)

    print(f"\n{'='*60}")
    print(f"  Norma   : {NORM_CODE}")
    print(f"  Tenant  : {tenant_id}")
    print(f"  Páginas : {PAGES_DIR}")
    print(f"  Estruturas alvo: {total}")
    print(f"  Modo    : {'DRY-RUN' if dry_run else 'REAL'}")
    print(f"{'='*60}\n")

    if not dry_run:
        print("  Limpando estruturas e materiais antigos...")
        delete_old(conn, NORM_ID)

    # Cria tarefas
    tasks = [(codigo, pages) for codigo, pages in PAGE_MAP.items()]

    results: dict[str, dict] = {}
    errors: list[str] = []
    t0 = time.monotonic()

    async def process(codigo, pages):
        result = await vision_extract_structure(codigo, pages, dry_run=dry_run)
        if result:
            results[codigo] = result
        else:
            errors.append(codigo)

    # Processa em lotes de MAX_PARALLEL
    for i in range(0, len(tasks), MAX_PARALLEL):
        batch = tasks[i:i + MAX_PARALLEL]
        pct = int((i / len(tasks)) * 100)
        bar = "#" * (pct // 5) + "." * (20 - pct // 5)
        print(f"  [{bar}] {pct:3d}% | {i}/{total} | {[c for c, _ in batch]}")
        await asyncio.gather(*[process(c, p) for c, p in batch])
        # Pausa entre lotes para respeitar rate limit
        if not dry_run and i + MAX_PARALLEL < len(tasks):
            await asyncio.sleep(3)

    elapsed = time.monotonic() - t0

    if not dry_run:
        # Salva no banco
        struct_count = 0
        mat_count = 0
        for codigo, data in results.items():
            sid = insert_structure(conn, NORM_ID, tenant_id, data)
            mats = data.get("materiais", [])
            if isinstance(mats, list):
                insert_materials(conn, NORM_ID, tenant_id, sid, codigo, mats)
                mat_count += len(mats)
            struct_count += 1

        # Atualiza status da norma
        conn.execute("""
            UPDATE technical_norms SET
                processing_status = 'done',
                processing_progress = 100,
                processing_message = ?,
                processing_finished_at = ?,
                pages_total = 266,
                pages_with_drawings = ?,
                pages_processed = ?
            WHERE id = ?
        """, (
            f"Indexação dirigida: {struct_count} estruturas, {mat_count} materiais",
            datetime.now(timezone.utc).isoformat(),
            total, struct_count, NORM_ID,
        ))
        conn.commit()

        print(f"\n{'='*60}")
        print(f"  CONCLUIDO")
        print(f"  Estruturas indexadas : {struct_count}/{total}")
        print(f"  Erros                : {len(errors)}")
        if errors:
            print(f"  Falhas               : {errors}")
        print(f"  Materiais salvos     : {mat_count}")
        print(f"  Tempo total          : {elapsed:.1f}s")
        print(f"{'='*60}\n")
    else:
        print(f"\n  [DRY-RUN] {len(results)}/{total} processados em {elapsed:.1f}s")

    conn.close()


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    asyncio.run(run(dry_run=dry))
