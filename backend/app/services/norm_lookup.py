"""Lookup de estruturas normativas para uso no Agente 01 (KMZ Analyzer).

Dado um código declarado no KMZ (ex: "SI3", "N1", "S4I"),
retorna a estrutura correspondente no banco + imagem do desenho técnico.
"""
from __future__ import annotations
import re
import sqlite3
from pathlib import Path

from loguru import logger

_DB_PATH = Path(__file__).resolve().parents[3] / "ops_ai_grid.db"
_PAGES_BASE = Path(__file__).resolve().parents[1] / "knowledge" / "pages"

# ID da norma NT.00022 (principal base de estruturas Equatorial)
NT00022_ID = "d06b92ba-37c9-4c45-8bf1-e41633674559"


def _normalize_variants(code: str) -> list[str]:
    """Retorna todas as grafias prováveis de um código de campo.

    Exemplos:
      "SI3"   → ["SI3", "S3I"]
      "S3I"   → ["S3I", "SI3"]
      "P3.P3" → ["P3.P3", "P3-P3"]
      "S4I-90°" → [..., "S4I-90"]
    """
    code = code.strip().upper()
    variants: set[str] = {code}

    # Troca SI{N} <-> S{N}I  (notação de campo vs norma)
    m = re.match(r"^SI(\d+)([A-Z]?)$", code)
    if m:
        variants.add(f"S{m.group(1)}I{m.group(2)}")
    m = re.match(r"^S(\d+)I([A-Z-]*)$", code)
    if m:
        variants.add(f"SI{m.group(1)}{m.group(2)}")

    # Pontos ↔ hífens  (P3.P3 = P3-P3)
    variants.add(code.replace(".", "-"))
    variants.add(code.replace("-", "."))

    # Remove símbolo de grau
    clean = code.replace("°", "").replace("°", "").replace("Â°", "")
    variants.add(clean)

    return [v for v in variants if v]


def _parse_codes_from_text(text: str) -> list[str]:
    """Extrai candidatos a códigos de estrutura de um texto livre.

    Padrão: token maiúsculo que contenha dígito OU tenha ≤3 chars com letra.
    Ex: 'PCC existente 32612254 SI4. SI3' → ['SI4', 'SI3']
    """
    # Remove HTML e tags especiais
    clean = re.sub(r"<[^>]+>", " ", text)
    # Captura tokens no formato código de estrutura
    tokens = re.findall(r"\b([A-Z][A-Z0-9]{1,4}(?:[-\.][A-Z0-9]{1,5})*)\b", clean.upper())

    IGNORE = {
        # Palavras de contexto / descritivas
        "PDT", "PCC", "PCT", "PMT", "PST",
        "MT", "BT", "AT", "DT", "CA", "PE", "KV",
        "KMZ", "KML", "GPS", "LAT", "LON", "ALT",
        "OK", "ID", "PRE", "IMG", "SRC", "MAX", "MIN", "COM",
        # Palavras portuguesas comuns
        "POSTE", "REDE", "RAMAL", "MALHA", "TERRA", "TIPO",
        "CABO", "FIOS", "CONCRETO", "ARMADO", "DUPLO",
        "MONO", "TRI", "BI", "FASE", "LINHA",
        "HASTE", "BLOCO", "PERFURANTE", "ALCA",
        "DE", "DO", "DA", "EM", "NO", "NA", "COM", "SEM",
        "POR", "UMA", "UM", "OS", "AS",
        # Unidades e abreviações
        "MM", "MT2", "KW", "KVA", "MVA",
    }

    result = []
    for t in dict.fromkeys(tokens):
        if t in IGNORE:
            continue
        if len(t) < 2:
            continue
        # Aceita se tem dígito (N1, SI3, S4I, U3) OU 2-3 chars (ex: futura expansão)
        has_digit = bool(re.search(r"\d", t))
        has_hyphen_or_dot = "-" in t or "." in t
        if has_digit or has_hyphen_or_dot or len(t) <= 3:
            result.append(t)

    return result


def parse_structure_codes(placemark_name: str, placemark_desc: str = "") -> list[str]:
    """Extrai códigos de estrutura do nome e descrição de um placemark KMZ.

    Retorna lista de candidatos ordenados por relevância (name primeiro).
    """
    # Prioriza o nome do placemark (mais confiável)
    from_name = _parse_codes_from_text(placemark_name)
    from_desc = _parse_codes_from_text(placemark_desc)
    # Une sem duplicatas, name primeiro
    seen: set[str] = set()
    result: list[str] = []
    for c in from_name + from_desc:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def lookup_structures(
    codes: list[str],
    norm_id: str | None = None,
) -> dict[str, dict]:
    """Busca estruturas normativas por código.

    Args:
        codes: Lista de códigos declarados no KMZ (ex: ["SI3", "N1"])
        norm_id: Filtra por norma específica (None = busca em todas)

    Returns:
        Dict código_original → row da norm_structures como dict.
        Inclui campo extra "_matched_as" com o código real no banco.
    """
    if not codes:
        return {}

    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    result: dict[str, dict] = {}

    try:
        c = conn.cursor()
        for code in codes:
            variants = _normalize_variants(code)
            placeholders = ",".join("?" * len(variants))
            q = (
                f"SELECT * FROM norm_structures "
                f"WHERE upper(codigo_estrutura) IN ({placeholders})"
            )
            params: list = [v.upper() for v in variants]
            if norm_id:
                q += " AND norm_id = ?"
                params.append(norm_id)
            q += " LIMIT 1"

            c.execute(q, params)
            row = c.fetchone()
            if row:
                d = dict(row)
                d["_matched_as"] = d["codigo_estrutura"]
                result[code] = d
                logger.debug(f"[norm_lookup] '{code}' → '{d['codigo_estrutura']}'")
            else:
                logger.debug(
                    f"[norm_lookup] '{code}' não encontrado "
                    f"(tentativas: {variants})"
                )
    finally:
        conn.close()

    return result


def get_drawing_bytes(struct: dict) -> bytes | None:
    """Carrega bytes JPEG do desenho técnico de uma estrutura.

    Tenta em ordem:
    1. imagem_desenho_path (path absoluto ou relativo a pages base)
    2. pagina_referencia + norm_id (path do renderizador)
    """
    img_path = struct.get("imagem_desenho_path", "")
    if img_path:
        p = Path(img_path)
        if p.is_file():
            return p.read_bytes()
        p2 = _PAGES_BASE / img_path
        if p2.is_file():
            return p2.read_bytes()

    page_num = struct.get("pagina_referencia", 0)
    norm_id = struct.get("norm_id", "")
    if page_num and norm_id:
        p3 = _PAGES_BASE / norm_id / f"page_{page_num:04d}.jpg"
        if p3.is_file():
            return p3.read_bytes()

    return None


def enrich_with_norm(
    placemark_name: str,
    placemark_desc: str = "",
    norm_id: str | None = None,
) -> dict:
    """Conveniência: parse + lookup em um passo.

    Returns dict com:
      - declared_codes: lista de todos os candidatos extraídos
      - found: dict code → struct (somente os encontrados no banco)
      - not_found: lista de códigos sem match
      - drawings: dict code → bytes JPEG (somente se imagem disponível)
    """
    candidates = parse_structure_codes(placemark_name, placemark_desc)
    found = lookup_structures(candidates, norm_id=norm_id or NT00022_ID)
    not_found = [c for c in candidates if c not in found]

    drawings: dict[str, bytes] = {}
    for code, struct in found.items():
        img = get_drawing_bytes(struct)
        if img:
            drawings[code] = img

    return {
        "declared_codes": candidates,
        "found": found,
        "not_found": not_found,
        "drawings": drawings,
    }
