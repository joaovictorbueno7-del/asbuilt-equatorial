"""Claude Vision wrapper specialized for Brazilian electrical distribution structures."""
from __future__ import annotations
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from dotenv import dotenv_values
from PIL import Image
from anthropic import AsyncAnthropic
from loguru import logger
from app.core.config import settings


def _resolve_api_key() -> str:
    """Settings/OS env can be empty-overridden by the shell. Read .env directly as fallback."""
    if settings.ANTHROPIC_API_KEY:
        return settings.ANTHROPIC_API_KEY
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        v = dotenv_values(str(env_path)).get("ANTHROPIC_API_KEY") or ""
        if v:
            return v
    return os.environ.get("ANTHROPIC_API_KEY", "")

VISION_PROMPT = """Voce e um especialista em redes de distribuicao de energia eletrica
brasileiras (padroes Equatorial / ANEEL). Analise a foto e identifique a estrutura
eletrica presente.

Retorne APENAS um JSON valido (nada antes ou depois) com este schema exato:
{
  "structure_type": "poste|transformador|chave|cruzeta|isolador|cabo|para_raios|medidor|outro",
  "condition": "boa|regular|ruim",
  "non_conformities": ["lista de problemas tecnicos visiveis"],
  "confidence": 0.0,
  "details": "descricao tecnica curta em portugues, ate 200 chars"
}

Regras:
- confidence entre 0.0 e 1.0
- non_conformities: itens curtos e tecnicos (ex: "isolador trincado", "vegetacao invasiva proxima ao cabo", "oxidacao no transformador")
- se a foto nao mostra estrutura eletrica, structure_type="outro" e confidence < 0.3
- nao escreva nada fora do JSON
"""

VISION_MODEL = "claude-sonnet-4-5"
MAX_DIMENSION = 1568

_client: AsyncAnthropic | None = None


def _client_or_raise() -> AsyncAnthropic:
    global _client
    if _client is None:
        key = _resolve_api_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured in .env")
        _client = AsyncAnthropic(api_key=key)
    return _client


def _normalize_image(raw: bytes) -> tuple[bytes, str]:
    """Re-encode to JPEG and downscale long edge to MAX_DIMENSION. Returns (bytes, media_type)."""
    img = Image.open(BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue(), "image/jpeg"


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["", "EXEMPLOS APROVADOS POR REVISORES (siga este padrao):"]
    for i, ex in enumerate(examples[:5], 1):
        out = ex.get("expected_output") or ex.get("observed_output") or {}
        if not out:
            continue
        lines.append(f"\nExemplo {i}:")
        lines.append(json.dumps({
            "structure_type": out.get("structure_type", "outro"),
            "condition": out.get("condition", "regular"),
            "non_conformities": out.get("non_conformities", []),
            "details": (out.get("details") or "")[:240],
        }, ensure_ascii=False))
    return "\n".join(lines)


async def analyze_image(raw_image: bytes, *, image_label: str = "",
                        examples: list[dict] | None = None) -> dict:
    """Send a single image to Claude Vision. Returns the parsed JSON dict
    or a safe fallback dict if anything fails. `examples` are LearningCases
    (with is_correct=True) used as few-shot."""
    label = image_label or "<image>"
    logger.info(f"[vision] analyze start: {label} ({len(raw_image)/1024:.1f}KB) "
                f"examples={len(examples or [])}")
    try:
        img_bytes, media_type = _normalize_image(raw_image)
    except Exception as e:
        logger.error(f"[vision] decode error {label}: {e}")
        return {
            "structure_type": "outro", "condition": "regular",
            "non_conformities": [], "confidence": 0.0,
            "details": f"image decode error: {type(e).__name__}",
        }

    client = _client_or_raise()
    b64 = base64.standard_b64encode(img_bytes).decode()
    prompt_text = VISION_PROMPT + _format_examples(examples or [])
    try:
        msg = await client.messages.create(
            model=VISION_MODEL,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt_text},
                ],
            }],
        )
    except Exception as e:
        logger.error(f"[vision] API error {label}: {e}")
        return {
            "structure_type": "outro", "condition": "regular",
            "non_conformities": [], "confidence": 0.0,
            "details": f"API error: {type(e).__name__}: {str(e)[:200]}",
        }
    text = "".join(getattr(b, "text", "") for b in msg.content)
    usage = getattr(msg, "usage", None)
    if usage:
        logger.info(f"[vision] {label} tokens in={usage.input_tokens} out={usage.output_tokens}")
    parsed = _extract_json(text)
    if parsed is None:
        logger.warning(f"[vision] {label} no JSON in response: {text[:100]}")
        return {
            "structure_type": "outro", "condition": "regular",
            "non_conformities": [], "confidence": 0.0,
            "details": "model did not return valid JSON",
            "raw_response": text[:300],
        }
    logger.info(f"[vision] {label} -> type={parsed.get('structure_type')} "
                f"cond={parsed.get('condition')} conf={parsed.get('confidence')}")

    parsed.setdefault("structure_type", "outro")
    parsed.setdefault("condition", "regular")
    parsed.setdefault("non_conformities", [])
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("details", "")
    try:
        parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0
    if not isinstance(parsed["non_conformities"], list):
        parsed["non_conformities"] = [str(parsed["non_conformities"])]
    return parsed
