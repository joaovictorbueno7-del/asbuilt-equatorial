"""Deep PDF reading: classify pages, send only the visually-rich ones to Claude Vision.

Strategy (Option C — hybrid):
1. Open PDF with PyMuPDF (fitz)
2. For each page extract text + count images + count vector drawings
3. Classify as "drawing page" if: contains structure markers ("Desenho N",
   "Estrutura X", "Lista de Materiais"), OR has images/many drawings AND short text
4. Drawing pages → render PNG @ 2x and send to Claude Vision with detailed prompt
5. Pure-text pages → skip (they're stored in TechnicalNorm.text_extracted already)
6. Aggregate structures + materials, dedupe by codigo_estrutura
"""
from __future__ import annotations
import asyncio
import base64
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
import fitz  # PyMuPDF
from PIL import Image
from loguru import logger
from anthropic import AsyncAnthropic
from agents.kmz_analyzer.vision import _resolve_api_key

VISION_MODEL = "claude-sonnet-4-5"
PARALLEL_PAGES = 4
RENDER_DPI = 144  # 2x of 72 = clear enough, ~150-300KB per page
MAX_IMAGE_DIMENSION = 1568

STRUCTURE_MARKER = re.compile(
    r"(Desenho\s+\d+|Estrutura\s+[A-Z]{1,5}[\-\.]?[A-Z]?\d*|Lista\s+de\s+Materiais|"
    r"Fixa[cç][aã]o\s+da\s+estrutura|PCC[\-\s]?[A-Z]{0,3}\d*|"
    r"\bM\-\d{2}\b|\bI\-\d{2}\b|\bF\-\d{2}\b)",
    re.IGNORECASE,
)


@dataclass
class PageClass:
    page_num: int          # 1-based
    text: str
    word_count: int
    image_count: int
    drawing_count: int
    has_marker: bool
    is_drawing_page: bool


@dataclass
class ExtractedPage:
    page_num: int
    structures: list[dict]
    materials: list[dict]
    raw_text: str = ""
    error: str = ""


def classify_pages(pdf_path: str) -> tuple[list[PageClass], "fitz.Document"]:
    doc = fitz.open(pdf_path)
    out: list[PageClass] = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        wc = len(text.split())
        img_count = len(page.get_images(full=True))
        drawing_count = len(page.get_drawings())
        has_marker = bool(STRUCTURE_MARKER.search(text))
        # heuristic: page likely shows a technical drawing if either
        is_drawing = (
            has_marker or
            (img_count > 0 and wc < 250) or
            (drawing_count > 30 and wc < 400)
        )
        out.append(PageClass(
            page_num=i + 1, text=text, word_count=wc,
            image_count=img_count, drawing_count=drawing_count,
            has_marker=has_marker, is_drawing_page=is_drawing,
        ))
    return out, doc


def render_page_png(doc: "fitz.Document", page_num: int, save_to: Path | None = None) -> bytes:
    page = doc[page_num - 1]
    matrix = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    raw_png = pix.tobytes("png")
    # Downscale for Claude (long edge <= MAX_IMAGE_DIMENSION) and re-encode JPEG
    img = Image.open(BytesIO(raw_png))
    if max(img.size) > MAX_IMAGE_DIMENSION:
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    out = buf.getvalue()
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_bytes(out)
    return out


PAGE_PROMPT = """Voce e um engenheiro eletricista experiente lendo uma norma tecnica
de uma concessionaria brasileira (Equatorial / similar). Esta pagina contem desenhos
tecnicos e tabelas de materiais.

Padrao tipico das normas Equatorial:
- Cabecalho: "Desenho X - [Tipo de Rede] - Estrutura [CODIGO]"
- Tabela "Lista de Materiais [CODIGO]" com colunas:
    Item (M-01, I-05, F-13...) | Tensao (23,1 / 34,5) | Codigo (134250015) | Quant. | Descricao
- Eventual segunda tabela "Fixacao da estrutura [CODIGO] no poste DT" com variacoes
  por comprimento (11m, 12/13m) e resistencia (300/600/1000/1500 daN)
- Codigos de estrutura: B4, S3I, PCC-SI4, DT-2, M1, etc.

Sua tarefa: extrair TUDO que ver na pagina como JSON valido (apenas o JSON, nada mais).

Schema:
{
  "structures": [
    {
      "codigo_estrutura": "PCC-SI4",
      "nome_completo": "Estrutura de Passagem com Cruzeta...",
      "tipo_rede": "MT|BT|AT",
      "tensao_nominal": "13,8kV",
      "desenho_numero": "5",
      "descricao_tecnica": "descricao tecnica curta (max 400 chars)",
      "caracteristicas_visuais": "como o desenho mostra a estrutura — geometria, partes",
      "como_identificar_na_foto": "o que procurar em foto de campo para identificar (max 300 chars)",
      "restricoes_uso": "limitacoes ou condicoes de aplicacao",
      "extraction_confidence": 0.0-1.0,
      "requires_review": false
    }
  ],
  "materials": [
    {
      "codigo_item": "M-01",
      "tensao": "23,1",
      "codigo_material": "134250015",
      "quantidade": "2",
      "descricao": "Abracadeira para poste...",
      "structure_codes": ["PCC-SI4"]
    }
  ],
  "fixacao": {
    "estrutura": "PCC-SI4",
    "variantes": [
      {"poste": "11m", "esforco": "300daN", "parafusos": {"P-02": 4, "P-08": 2}}
    ]
  }
}

Regras:
- Se nao ha estrutura/material visivel: retorne {"structures": [], "materials": []}
- structures e materials sao listas; "fixacao" e opcional (omitir se nao houver tabela de fixacao)
- materials.structure_codes liga o material a uma ou mais estruturas vistas na pagina
- extraction_confidence: 0.9+ se voce esta certo do codigo e ve a tabela completa; 0.5- se duvidoso
- requires_review = true quando codigo da estrutura esta truncado, ilegivel ou voce nao consegue identificar
- nao invente codigos, materiais ou medidas que nao estao na pagina
"""


_client: AsyncAnthropic | None = None


def _client_or_raise() -> AsyncAnthropic:
    global _client
    if _client is None:
        key = _resolve_api_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        _client = AsyncAnthropic(api_key=key)
    return _client


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def vision_extract_page(page_image_jpeg: bytes, page_num: int,
                               surrounding_text: str = "") -> ExtractedPage:
    client = _client_or_raise()
    b64 = base64.standard_b64encode(page_image_jpeg).decode()
    prompt = PAGE_PROMPT
    if surrounding_text:
        prompt += "\n\nTexto OCR/Pyumpdf desta pagina (apoio):\n" + surrounding_text[:2000]
    try:
        msg = await client.messages.create(
            model=VISION_MODEL,
            max_tokens=2500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
    except Exception as e:
        logger.error(f"[deep-extractor] page {page_num} API error: {e}")
        return ExtractedPage(page_num=page_num, structures=[], materials=[],
                              error=f"API: {type(e).__name__}: {str(e)[:200]}")
    text = "".join(getattr(b, "text", "") for b in msg.content)
    usage = getattr(msg, "usage", None)
    if usage:
        logger.info(f"[deep-extractor] page {page_num} tokens in={usage.input_tokens} out={usage.output_tokens}")
    parsed = _extract_json(text)
    if not parsed:
        logger.warning(f"[deep-extractor] page {page_num} no JSON: {text[:120]}")
        return ExtractedPage(page_num=page_num, structures=[], materials=[],
                              error="no JSON in model response", raw_text=text[:300])
    structs = parsed.get("structures", [])
    mats = parsed.get("materials", [])
    fixacao = parsed.get("fixacao") or {}
    if fixacao and isinstance(fixacao, dict) and structs:
        for s in structs:
            if s.get("codigo_estrutura") == fixacao.get("estrutura"):
                s["fixacao"] = fixacao
    return ExtractedPage(
        page_num=page_num,
        structures=structs if isinstance(structs, list) else [],
        materials=mats if isinstance(mats, list) else [],
    )


@dataclass
class DeepResult:
    pages_total: int
    pages_drawing: int
    pages_processed: int
    structures: list[dict]
    materials_by_code: dict[str, dict]   # codigo_material -> dict
    page_errors: list[tuple[int, str]]
    cost_pages_visioned: int


def _aggregate_materials(materials: list[dict], by_code: dict[str, dict],
                         page_num: int):
    for m in materials:
        codigo = (m.get("codigo_material") or m.get("codigo_item") or "").strip()
        if not codigo:
            continue
        bucket = by_code.setdefault(codigo, {
            "codigo_material": codigo,
            "codigo_item": (m.get("codigo_item") or "").strip(),
            "descricao": (m.get("descricao") or "").strip(),
            "tensao": (m.get("tensao") or "").strip(),
            "structure_codes": set(),
            "first_seen_page": page_num,
        })
        # backfill best info
        if not bucket["descricao"] and m.get("descricao"):
            bucket["descricao"] = m["descricao"].strip()
        if not bucket["codigo_item"] and m.get("codigo_item"):
            bucket["codigo_item"] = m["codigo_item"].strip()
        for sc in m.get("structure_codes") or []:
            bucket["structure_codes"].add(str(sc).strip())


def _aggregate_structures(structures: list[dict], page_num: int,
                          existing_by_code: dict[str, dict],
                          image_path: str | None):
    """Dedupe by codigo_estrutura. Keep first-seen + merge missing fields."""
    for s in structures:
        codigo = (s.get("codigo_estrutura") or "").strip()
        if not codigo:
            continue
        if codigo in existing_by_code:
            # merge non-empty fields
            existing = existing_by_code[codigo]
            for k in ("descricao_tecnica", "caracteristicas_visuais",
                       "como_identificar_na_foto", "restricoes_uso",
                       "tipo_rede", "tensao_nominal", "desenho_numero"):
                if not existing.get(k) and s.get(k):
                    existing[k] = s[k]
            continue
        s_clean = dict(s)
        s_clean["pagina_referencia"] = page_num
        s_clean["imagem_desenho_path"] = image_path or ""
        existing_by_code[codigo] = s_clean


async def deep_process_pdf(pdf_path: str, norm_id: str,
                           pages_dir: Path,
                           progress_callback) -> DeepResult:
    """Top-level orchestrator. progress_callback(step, current, total, message) is async."""
    await progress_callback("classifying", 0, 0, "abrindo PDF")
    pages_info, doc = classify_pages(pdf_path)
    total = len(pages_info)
    drawing_pages = [p for p in pages_info if p.is_drawing_page]
    n_draw = len(drawing_pages)
    logger.info(f"[deep-extractor] {norm_id}: {total} pages, {n_draw} flagged as drawing")
    await progress_callback("extracting", 0, n_draw,
                             f"{n_draw} de {total} paginas serao processadas com Vision")

    sem = asyncio.Semaphore(PARALLEL_PAGES)
    structures_by_code: dict[str, dict] = {}
    materials_by_code: dict[str, dict] = {}
    page_errors: list[tuple[int, str]] = []
    processed = 0

    async def worker(page_class: PageClass):
        nonlocal processed
        page_image_path = pages_dir / f"page_{page_class.page_num:04d}.jpg"
        try:
            img_bytes = render_page_png(doc, page_class.page_num, save_to=page_image_path)
        except Exception as e:
            logger.error(f"[deep-extractor] page {page_class.page_num} render error: {e}")
            page_errors.append((page_class.page_num, f"render: {e}"))
            return
        async with sem:
            result = await vision_extract_page(
                img_bytes, page_class.page_num,
                surrounding_text=page_class.text[:1500],
            )
        if result.error:
            page_errors.append((page_class.page_num, result.error))
        else:
            _aggregate_structures(result.structures, page_class.page_num,
                                   structures_by_code, str(page_image_path))
            _aggregate_materials(result.materials, materials_by_code,
                                  page_class.page_num)
        processed += 1
        await progress_callback("extracting", processed, n_draw,
                                 f"pagina {page_class.page_num}/{total}")

    if n_draw > 0:
        await asyncio.gather(*[worker(p) for p in drawing_pages])

    doc.close()

    # convert sets to sorted lists for JSON-friendliness
    for m in materials_by_code.values():
        m["structure_codes"] = sorted(m["structure_codes"])

    return DeepResult(
        pages_total=total,
        pages_drawing=n_draw,
        pages_processed=processed,
        structures=list(structures_by_code.values()),
        materials_by_code=materials_by_code,
        page_errors=page_errors,
        cost_pages_visioned=processed,
    )
