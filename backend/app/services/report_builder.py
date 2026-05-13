"""Gerador de Relatório de Entrega de Obra — formato Equatorial.

Produz DOCX (python-docx) e PDF (reportlab) no layout oficial:
  - Cabeçalho em todas as páginas: logo esq | título centralizado | logo dir | linha
  - Página 1: metadados (nota, município, parceira) + título "Postes, Estruturas e Redes"
  - Grade de fotos: 2 por linha, legenda em negrito, títulos de grupo
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Any

from loguru import logger

# ── python-docx ─────────────────────────────────────────────────────────────
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── reportlab ────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_STORAGE = Path(os.environ.get("STORAGE_LOCAL_PATH", "storage"))
_REPORTS_DIR = _STORAGE / "reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Tamanho máximo de imagem (evita fotos absurdamente grandes)
_MAX_IMG_BYTES = 8 * 1024 * 1024  # 8 MB


def _load_image_bytes(src: str | None, kmz_path: str | None) -> bytes | None:
    """Carrega bytes de uma imagem: caminho absoluto ou dentro do KMZ."""
    if not src:
        return None
    # Caminho absoluto
    p = Path(src)
    if p.is_file():
        data = p.read_bytes()
        return data if len(data) <= _MAX_IMG_BYTES else None
    # Dentro do KMZ
    if kmz_path and Path(kmz_path).is_file():
        safe = src.replace("..", "").lstrip("/").lstrip("\\")
        try:
            with zipfile.ZipFile(kmz_path, "r") as zf:
                names = zf.namelist()
                target = next(
                    (n for n in names if n == safe or Path(n).name == Path(safe).name),
                    None,
                )
                if target:
                    return zf.read(target)
        except Exception:
            pass
    return None


def _resize_image(data: bytes, max_w: int = 800, max_h: int = 600) -> bytes:
    """Redimensiona imagem se necessária (usando Pillow)."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = img.format or "JPEG"
        if fmt not in ("JPEG", "PNG"):
            fmt = "JPEG"
            img = img.convert("RGB")
        img.save(buf, format=fmt, quality=85)
        return buf.getvalue()
    except Exception:
        return data


def _group_structures(structures: list[dict]) -> list[tuple[str, list[dict]]]:
    """Agrupa estruturas por tipo de poste para seções do relatório.

    Retorna lista de (título_grupo, [estruturas]).
    """
    groups: dict[str, list[dict]] = {}
    for s in structures:
        # Usa nome do placemark ou tipo de estrutura como chave do grupo
        pname = s.get("placemark_name", "")
        # Extrai prefixo do código (ex: "PDT 9/300" de "PDT 9/300. 47023256 SI3")
        parts = pname.split(".")
        group_key = parts[0].strip() if parts else pname
        if not group_key:
            group_key = s.get("structure_type", "Outras Estruturas")
        groups.setdefault(group_key, []).append(s)
    return list(groups.items())


# ─────────────────────────────────────────────────────────────────────────────
# DOCX Builder
# ─────────────────────────────────────────────────────────────────────────────

def _set_cell_border(cell, **kwargs):
    """Define bordas de uma célula DOCX via XML (helper)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "start", "bottom", "end", "insideH", "insideV"):
        val = kwargs.get(edge, "none")
        tag = OxmlElement(f"w:{edge}")
        tag.set(qn("w:val"), val)
        tag.set(qn("w:sz"), "4")
        tag.set(qn("w:color"), "000000")
        tcBorders.append(tag)
    tcPr.append(tcBorders)


def _add_horizontal_rule(doc: Document, thickness_pt: float = 1.0):
    """Adiciona linha horizontal no documento DOCX."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(int(thickness_pt * 8)))
    bottom.set(qn("w:color"), "404040")
    pBdr.append(bottom)
    pPr.append(pBdr)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    return p


def _add_header(doc: Document, tipo: str = "Relatório de Entrega de Obra"):
    """Adiciona cabeçalho padrão Equatorial em todas as seções."""
    section = doc.sections[0]
    header = section.header
    header.is_linked_to_previous = False

    # Tabela 1x3: logo | título | logo parceira
    htable = header.add_table(rows=1, cols=3, width=Inches(6.5))
    htable.alignment = WD_TABLE_ALIGNMENT.CENTER
    htable.style = "Table Grid"

    # Célula esquerda: "EQUATORIAL" em azul (placeholder sem imagem real)
    lc = htable.cell(0, 0)
    lc.width = Inches(1.5)
    lc.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    lp = lc.paragraphs[0]
    lp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    lr = lp.add_run("EQUATORIAL")
    lr.bold = True
    lr.font.size = Pt(10)
    lr.font.color.rgb = RGBColor(0x00, 0x5F, 0xAF)

    # Célula central: título do relatório
    mc = htable.cell(0, 1)
    mc.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    mp = mc.paragraphs[0]
    mp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = mp.add_run("Relatório de Entrega de Obra")
    mr.bold = True
    mr.font.size = Pt(9)
    # Segundo parágrafo: tipo
    mp2 = mc.add_paragraph()
    mp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mp2.add_run(tipo).font.size = Pt(8)
    # Terceiro parágrafo: regional
    mp3 = mc.add_paragraph()
    mp3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = mp3.add_run("Superintendência Centro – Regional Metropolitana")
    r3.font.size = Pt(7)
    r3.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    # Célula direita: nome da parceira / logo placeholder
    rc = htable.cell(0, 2)
    rc.width = Inches(1.5)
    rc.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    rp = rc.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rr = rp.add_run("OPS AI GRID")
    rr.font.size = Pt(8)
    rr.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Remove bordas da tabela do cabeçalho
    for row in htable.rows:
        for cell in row.cells:
            _set_cell_border(cell)

    # Linha separadora abaixo da tabela
    sep_p = header.add_paragraph()
    sep_p.paragraph_format.space_before = Pt(2)
    sep_p.paragraph_format.space_after = Pt(0)
    pPr = sep_p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:color"), "005FAF")
    pBdr.append(bottom)
    pPr.append(pBdr)


def build_docx(
    run_id: str,
    metadata: dict,
    structures: list[dict],
    kmz_path: str | None = None,
) -> bytes:
    """Constrói DOCX no formato Equatorial.

    Args:
        run_id: ID do AgentRun (para logs)
        metadata: {nota, municipio, parceira, tipo}
        structures: lista de dicts com {placemark_name, caption, image_src, ...}
        kmz_path: caminho para o arquivo .kmz (para extrair fotos)

    Returns:
        bytes do arquivo .docx
    """
    nota = metadata.get("nota", "—")
    municipio = metadata.get("municipio", "—")
    parceira = metadata.get("parceira", "—")
    tipo = metadata.get("tipo", "Postes, Estruturas e Redes")

    doc = Document()

    # ── Configuração de página ────────────────────────────────────────────────
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(3.0)
    section.bottom_margin = Cm(2.0)

    # ── Cabeçalho ────────────────────────────────────────────────────────────
    _add_header(doc, tipo)

    # ── Página 1: metadados ───────────────────────────────────────────────────
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"Nota: {nota}")
    r.font.size = Pt(11)

    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(2)
    r2 = p2.add_run(f"Município: {municipio}")
    r2.font.size = Pt(11)

    p3 = doc.add_paragraph()
    p3.paragraph_format.space_after = Pt(8)
    r3 = p3.add_run(f"Parceira Construção: {parceira}")
    r3.font.size = Pt(11)

    # Título centralizado em negrito
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(12)
    rt = p_title.add_run("Postes, Estruturas e Redes")
    rt.bold = True
    rt.font.size = Pt(13)

    # ── Grade de fotos ────────────────────────────────────────────────────────
    groups = _group_structures(structures)
    usable_width = section.page_width - section.left_margin - section.right_margin
    # Largura de cada foto: ~45% do espaço útil
    photo_w_emu = int(usable_width * 0.45)
    photo_w_inches = photo_w_emu / 914400  # EMU → inches

    for group_title, group_structs in groups:
        # Título do grupo (negrito, centralizado)
        gp = doc.add_paragraph()
        gp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        gp.paragraph_format.space_before = Pt(8)
        gp.paragraph_format.space_after = Pt(4)
        gr = gp.add_run(group_title)
        gr.bold = True
        gr.font.size = Pt(11)

        # Processa fotos em pares
        for i in range(0, len(group_structs), 2):
            pair = group_structs[i: i + 2]
            # Tabela 2-colunas para o par de fotos
            tbl = doc.add_table(rows=2, cols=len(pair))
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            tbl.style = "Table Grid"

            for col_idx, struct in enumerate(pair):
                img_src = struct.get("image_src") or struct.get("photo_src")
                caption = struct.get("caption") or struct.get("placemark_name") or "—"

                # Linha 0: foto
                cell_img = tbl.cell(0, col_idx)
                cell_img.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                _set_cell_border(cell_img)
                cp = cell_img.paragraphs[0]
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cp.paragraph_format.space_before = Pt(2)
                cp.paragraph_format.space_after = Pt(2)

                img_bytes = _load_image_bytes(img_src, kmz_path)
                if img_bytes:
                    try:
                        img_bytes = _resize_image(img_bytes)
                        img_stream = io.BytesIO(img_bytes)
                        cp.add_run().add_picture(img_stream, width=Inches(photo_w_inches))
                    except Exception as e:
                        logger.warning(f"[report_builder] foto falhou ({img_src}): {e}")
                        cp.add_run("[foto indisponível]").font.size = Pt(9)
                else:
                    cp.add_run("[foto indisponível]").font.size = Pt(9)

                # Linha 1: legenda em negrito
                cell_cap = tbl.cell(1, col_idx)
                _set_cell_border(cell_cap)
                lp = cell_cap.paragraphs[0]
                lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                lp.paragraph_format.space_before = Pt(2)
                lp.paragraph_format.space_after = Pt(2)
                lr = lp.add_run(caption)
                lr.bold = True
                lr.font.size = Pt(9)

            # Se só 1 estrutura no par, mescla coluna vazia (só aparece se colunas=1)
            # (já tratado automaticamente com len(pair) colunas)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # ── Salva em bytes ─────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# PDF Builder (reportlab)
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf(
    run_id: str,
    metadata: dict,
    structures: list[dict],
    kmz_path: str | None = None,
) -> bytes:
    """Constrói PDF no formato Equatorial usando reportlab.

    Mesma estrutura do DOCX.
    """
    nota = metadata.get("nota", "—")
    municipio = metadata.get("municipio", "—")
    parceira = metadata.get("parceira", "—")
    tipo = metadata.get("tipo", "Postes, Estruturas e Redes")

    buf = io.BytesIO()
    PAGE_W, PAGE_H = A4  # 595 × 842 pt
    margin = 2 * cm
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=3.5 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    # Estilos customizados
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, leading=16,
                         alignment=TA_CENTER, spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, leading=14,
                         alignment=TA_CENTER, spaceAfter=6, spaceBefore=10)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11, leading=14,
                           spaceAfter=4)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9,
                                    leading=12, alignment=TA_CENTER, fontName="Helvetica-Bold")
    header_center = ParagraphStyle("HdrCenter", parent=styles["Normal"], fontSize=8,
                                    alignment=TA_CENTER, leading=11)
    header_right = ParagraphStyle("HdrRight", parent=styles["Normal"], fontSize=8,
                                   alignment=TA_RIGHT)

    content = []

    # ── Cabeçalho (via canvas callback) ──────────────────────────────────────
    equatorial_label = "EQUATORIAL"
    title_lines = [
        "Relatório de Entrega de Obra",
        tipo,
        "Superintendência Centro – Regional Metropolitana",
    ]
    partner_label = "OPS AI GRID"

    def on_first_page(canvas, doc):
        _draw_pdf_header(canvas, PAGE_W, PAGE_H, doc.topMargin,
                         equatorial_label, title_lines, partner_label)

    def on_later_pages(canvas, doc):
        _draw_pdf_header(canvas, PAGE_W, PAGE_H, doc.topMargin,
                         equatorial_label, title_lines, partner_label)

    # ── Metadados ─────────────────────────────────────────────────────────────
    content.append(Paragraph(f"<b>Nota:</b> {nota}", body))
    content.append(Paragraph(f"<b>Município:</b> {municipio}", body))
    content.append(Paragraph(f"<b>Parceira Construção:</b> {parceira}", body))
    content.append(Spacer(1, 12))
    content.append(Paragraph("Postes, Estruturas e Redes", h1))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#005FAF")))
    content.append(Spacer(1, 12))

    # ── Grade de fotos ────────────────────────────────────────────────────────
    photo_w = (PAGE_W - 2 * margin - 1 * cm) * 0.47  # ~47% de largura
    photo_h = photo_w * 0.75  # proporção 4:3

    groups = _group_structures(structures)
    for group_title, group_structs in groups:
        content.append(Paragraph(group_title, h2))

        for i in range(0, len(group_structs), 2):
            pair = group_structs[i: i + 2]
            row_imgs = []
            row_captions = []

            for struct in pair:
                img_src = struct.get("image_src") or struct.get("photo_src")
                caption = struct.get("caption") or struct.get("placemark_name") or "—"
                img_bytes = _load_image_bytes(img_src, kmz_path)

                if img_bytes:
                    try:
                        img_bytes = _resize_image(img_bytes, 800, 600)
                        rl_img = RLImage(io.BytesIO(img_bytes), width=photo_w, height=photo_h)
                        row_imgs.append(rl_img)
                    except Exception as e:
                        logger.warning(f"[report_builder/pdf] foto falhou: {e}")
                        row_imgs.append(Paragraph("[foto indisponível]", caption_style))
                else:
                    row_imgs.append(Paragraph("[foto indisponível]", caption_style))

                row_captions.append(Paragraph(caption, caption_style))

            # Preenche segunda célula se par incompleto
            if len(pair) == 1:
                row_imgs.append(Spacer(photo_w, photo_h))
                row_captions.append(Paragraph("", caption_style))

            col_w = (PAGE_W - 2 * margin) / 2
            tbl = Table(
                [row_imgs, row_captions],
                colWidths=[col_w, col_w],
            )
            tbl.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                ("VALIGN", (0, 1), (-1, 1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("TOPPADDING", (0, 1), (-1, 1), 2),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            content.append(tbl)
            content.append(Spacer(1, 8))

    doc.build(content, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    return buf.getvalue()


def _draw_pdf_header(canvas, page_w, page_h, top_margin, equatorial, title_lines, partner):
    """Desenha cabeçalho fixo em cada página PDF."""
    from reportlab.lib.units import cm as rl_cm
    canvas.saveState()

    hdr_y = page_h - top_margin + 0.5 * rl_cm
    hdr_h = 1.8 * rl_cm
    margin = 2 * rl_cm
    usable_w = page_w - 2 * margin

    # Fundo do cabeçalho (levemente cinza)
    canvas.setFillColorRGB(0.97, 0.97, 0.97)
    canvas.rect(margin, hdr_y - hdr_h, usable_w, hdr_h, fill=1, stroke=0)

    # Logo esquerda (texto azul em negrito)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColorRGB(0.0, 0.37, 0.69)
    canvas.drawString(margin + 4, hdr_y - 14, equatorial)

    # Título centralizado
    canvas.setFillColorRGB(0.1, 0.1, 0.1)
    cx = page_w / 2
    if len(title_lines) >= 1:
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawCentredString(cx, hdr_y - 12, title_lines[0])
    if len(title_lines) >= 2:
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(cx, hdr_y - 22, title_lines[1])
    if len(title_lines) >= 3:
        canvas.setFont("Helvetica", 7)
        canvas.setFillColorRGB(0.4, 0.4, 0.4)
        canvas.drawCentredString(cx, hdr_y - 32, title_lines[2])

    # Parceira (direita)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColorRGB(0.4, 0.4, 0.4)
    canvas.drawRightString(page_w - margin - 4, hdr_y - 14, partner)

    # Linha azul separadora
    canvas.setStrokeColorRGB(0.0, 0.37, 0.69)
    canvas.setLineWidth(1.5)
    canvas.line(margin, hdr_y - hdr_h - 1, page_w - margin, hdr_y - hdr_h - 1)

    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Extração de estruturas a partir do output do AgentRun
# ─────────────────────────────────────────────────────────────────────────────

def extract_structures_from_run(output_payload: dict, kmz_path: str | None = None) -> list[dict]:
    """Converte o output_payload do AgentRun em lista de estruturas para o relatório.

    Suporta formato do kmz_analyzer (placemarks com fotos) e do description_filler.
    """
    structures = []

    # Formato 1: placemarks do kmz_analyzer
    placemarks = output_payload.get("placemarks") or output_payload.get("structures") or []
    for pm in placemarks:
        name = pm.get("name") or pm.get("placemark_name") or "—"
        photos = pm.get("photos") or pm.get("images") or []

        # Monta legenda: "Poste DT 11/300 - N1 SI3"
        struct_codes = []
        if pm.get("conformidade"):
            confirmed = pm.get("estruturas_confirmadas") or pm.get("declared_codes") or []
            struct_codes = confirmed[:2] if confirmed else []
        elif pm.get("structure_type"):
            struct_codes = [pm["structure_type"]]

        # Extrai tipo de poste do nome (ex: "PDT 9/300")
        poste_type = _extract_poste_type(name)
        codes_str = " ".join(struct_codes) if struct_codes else name

        if photos:
            for photo in photos:
                photo_src = photo if isinstance(photo, str) else photo.get("src") or photo.get("path", "")
                caption = f"{poste_type} - {codes_str}".strip(" -")
                structures.append({
                    "placemark_name": name,
                    "image_src": photo_src,
                    "caption": caption,
                })
        else:
            # Sem foto: inclui mesmo assim (mostrará placeholder)
            caption = f"{poste_type} - {codes_str}".strip(" -")
            structures.append({
                "placemark_name": name,
                "image_src": None,
                "caption": caption,
            })

    return structures


def _extract_poste_type(name: str) -> str:
    """Extrai tipo do poste do nome do placemark (ex: 'PDT 9/300')."""
    import re
    # Padrão: PDT 9/300, PCC 9/300, DT 11/300, etc.
    m = re.search(r"\b(P?[A-Z]{2,3})\s+(\d+/\d+)", name, re.IGNORECASE)
    if m:
        return f"Poste {m.group(1).upper()} {m.group(2)}"
    m2 = re.search(r"\b([A-Z]{2,3})\s+(\d+)", name, re.IGNORECASE)
    if m2:
        return f"Poste {m2.group(1).upper()} {m2.group(2)}"
    return "Poste"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    run_id: str,
    output_payload: dict,
    kmz_path: str | None = None,
    fmt: str = "docx",
) -> tuple[bytes, str]:
    """Gera relatório a partir do output de um AgentRun.

    Args:
        run_id: ID do run
        output_payload: dict com placemarks/structures
        kmz_path: caminho do KMZ (para extração de fotos)
        fmt: "docx" ou "pdf"

    Returns:
        (bytes_do_arquivo, nome_do_arquivo)
    """
    metadata = {
        "nota": output_payload.get("nota") or output_payload.get("work_name") or run_id[:8],
        "municipio": output_payload.get("municipio") or "—",
        "parceira": output_payload.get("parceira") or output_payload.get("concessionaria") or "—",
        "tipo": output_payload.get("tipo") or "Postes, Estruturas e Redes",
    }

    structures = extract_structures_from_run(output_payload, kmz_path)
    logger.info(f"[report_builder] run={run_id} fmt={fmt} estruturas={len(structures)}")

    if fmt == "pdf":
        data = build_pdf(run_id, metadata, structures, kmz_path)
        filename = f"relatorio_{run_id[:8]}.pdf"
    else:
        data = build_docx(run_id, metadata, structures, kmz_path)
        filename = f"relatorio_{run_id[:8]}.docx"

    # Salva cópia em disco
    out_path = _REPORTS_DIR / filename
    out_path.write_bytes(data)
    logger.info(f"[report_builder] salvo em {out_path} ({len(data):,} bytes)")

    return data, filename
