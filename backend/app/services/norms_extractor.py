"""Extract text from norm PDFs and ask Claude to structure into NormStructure rows."""
from __future__ import annotations
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader
from loguru import logger
from anthropic import AsyncAnthropic
from agents.kmz_analyzer.vision import _resolve_api_key

EXTRACT_MODEL = "claude-sonnet-4-5"
MAX_PROMPT_CHARS = 60_000  # truncate very large PDFs

EXTRACTION_PROMPT = """Voce esta lendo o texto extraido de uma norma tecnica de uma
concessionaria de energia eletrica brasileira. Identifique as ESTRUTURAS ELETRICAS
documentadas (postes, transformadores, chaves, cruzetas, isoladores, etc.) e retorne
APENAS um JSON valido com este schema:

{
  "structures": [
    {
      "codigo_estrutura": "PT-01",
      "nome_completo": "Poste tipo I de concreto duplo T",
      "descricao_tecnica": "descricao tecnica curta (max 400 chars)",
      "caracteristicas_visuais": "como identificar visualmente em foto",
      "campos_proj": {
        "tipo": "string", "altura_m": "number", "esforco_dan": "number"
      },
      "materiais": ["lista", "de", "materiais", "com", "quantidade"]
    }
  ]
}

Regras:
- so estruturas eletricas (nao tabelas administrativas)
- codigo_estrutura: use o codigo oficial da norma (ex: "PT-01", "TR-014")
- campos_proj: chaves sao os campos a preencher no sistema PROJ; valores sao tipos
- nao inventar dados que nao estao no texto
- max 50 estruturas por chamada
- nada fora do JSON"""


def md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def extract_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Returns (full_text, page_count). Best-effort extraction; returns empty if PDF unreadable."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception as e:
                logger.warning(f"[norms] page extract error: {e}")
        return "\n\n".join(pages), len(reader.pages)
    except Exception as e:
        logger.error(f"[norms] PDF parse failed: {e}")
        return "", 0


async def claude_structure(text: str) -> list[dict]:
    """Send extracted text to Claude and parse out structures list."""
    if not text.strip():
        return []
    key = _resolve_api_key()
    if not key:
        logger.warning("[norms] no ANTHROPIC_API_KEY; skipping structuring")
        return []
    client = AsyncAnthropic(api_key=key)
    truncated = text[:MAX_PROMPT_CHARS]
    if len(text) > MAX_PROMPT_CHARS:
        logger.warning(f"[norms] truncated text from {len(text)} to {MAX_PROMPT_CHARS} chars")
    try:
        msg = await client.messages.create(
            model=EXTRACT_MODEL,
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "text", "text": "\n\n--- TEXTO DA NORMA ---\n\n" + truncated},
                ],
            }],
        )
    except Exception as e:
        logger.error(f"[norms] Claude API error: {e}")
        return []
    raw = "".join(getattr(b, "text", "") for b in msg.content)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        logger.warning(f"[norms] no JSON in response: {raw[:200]}")
        return []
    try:
        parsed = json.loads(m.group(0))
        out = parsed.get("structures", [])
        logger.info(f"[norms] extracted {len(out)} structures")
        return out if isinstance(out, list) else []
    except json.JSONDecodeError as e:
        logger.error(f"[norms] JSON decode: {e}")
        return []
