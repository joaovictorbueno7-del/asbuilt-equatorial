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

# ─── Prompt de comparação campo x norma ────────────────────────────────────
COMPARISON_PROMPT = """Você é auditor técnico especializado em redes de distribuição elétrica (normas Equatorial NT.00022).

CONTEXTO DO CAMPO:
- Poste/ponto: {poste_info}
- Estruturas DECLARADAS no KMZ: {estruturas_str}

A sequência de imagens é:
1. FOTO DO CAMPO (primeira imagem)
{drawings_list}

Compare a FOTO DO CAMPO com os DESENHOS TÉCNICOS das normas e retorne um JSON:
{{
  "poste": "identificação do poste conforme KMZ",
  "estruturas_declaradas": {estruturas_json},
  "estruturas_confirmadas": ["estruturas visualmente presentes e compatíveis com a norma"],
  "estruturas_divergentes": ["declaradas no KMZ mas incompatíveis ou ausentes na foto"],
  "materiais_visiveis": ["materiais claramente identificáveis na foto"],
  "materiais_faltantes": ["materiais esperados pela norma mas não visíveis na foto"],
  "conformidade": true,
  "confianca": 0.0,
  "observacoes": "observações técnicas em português (máx 300 chars)"
}}

Regras:
- conformidade=true somente se estruturas_divergentes=[] E materiais_faltantes tem ≤1 item
- confianca entre 0.0 e 1.0 (certeza baseada na qualidade da foto e visibilidade dos componentes)
- Se a foto for de baixa qualidade/ângulo ruim, reduza confianca
- estruturas_confirmadas: inclua APENAS o que você identifica com certeza visual
- Retorne APENAS o JSON, nada antes ou depois"""


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


async def compare_with_norm(
    field_photo: bytes,
    *,
    declared_codes: list[str],
    norm_structs: dict[str, dict],       # code → norm_structures row
    drawing_images: dict[str, bytes],    # code → JPEG bytes
    poste_info: str = "",
    image_label: str = "",
) -> dict:
    """Compara foto de campo com desenhos técnicos das normas.

    Envia para Claude Vision:
      - Foto do campo
      - Desenho técnico de cada estrutura declarada (se disponível)
    Retorna JSON com conformidade, divergências e materiais.
    """
    label = image_label or "campo"
    try:
        field_bytes, field_media = _normalize_image(field_photo)
    except Exception as e:
        logger.error(f"[vision-compare] decode field photo {label}: {e}")
        return _fallback_compare(declared_codes, poste_info, f"decode error: {e}")

    # Constrói conteúdo: foto do campo + desenhos técnicos
    content: list[dict] = [
        {"type": "image", "source": {
            "type": "base64", "media_type": field_media,
            "data": base64.standard_b64encode(field_bytes).decode()}},
        {"type": "text", "text": "↑ FOTO DO CAMPO"},
    ]

    drawings_list_lines: list[str] = []
    idx = 2
    for code in declared_codes:
        raw_drw = drawing_images.get(code)
        if raw_drw:
            try:
                drw_bytes, drw_media = _normalize_image(raw_drw)
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": drw_media,
                    "data": base64.standard_b64encode(drw_bytes).decode()}})
                nome = norm_structs.get(code, {}).get("nome_completo", code)
                content.append({"type": "text", "text": f"↑ DESENHO TÉCNICO NORMA: {code} — {nome}"})
                drawings_list_lines.append(f"{idx}. DESENHO TÉCNICO da estrutura {code}")
                idx += 1
            except Exception as e:
                logger.warning(f"[vision-compare] drawing {code} error: {e}")
        else:
            drawings_list_lines.append(f"(sem desenho disponível para {code})")

    drawings_list = "\n".join(drawings_list_lines) if drawings_list_lines else "(nenhum desenho técnico disponível)"

    prompt = COMPARISON_PROMPT.format(
        poste_info=poste_info or "não especificado",
        estruturas_str=", ".join(declared_codes) if declared_codes else "não especificadas",
        drawings_list=drawings_list,
        estruturas_json=json.dumps(declared_codes, ensure_ascii=False),
    )
    content.append({"type": "text", "text": prompt})

    client = _client_or_raise()
    try:
        msg = await client.messages.create(
            model=VISION_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:
        logger.error(f"[vision-compare] API error {label}: {e}")
        return _fallback_compare(declared_codes, poste_info, str(e)[:200])

    text = "".join(getattr(b, "text", "") for b in msg.content)
    usage = getattr(msg, "usage", None)
    if usage:
        logger.info(f"[vision-compare] {label} tokens in={usage.input_tokens} out={usage.output_tokens}")

    parsed = _extract_json(text)
    if parsed is None:
        logger.warning(f"[vision-compare] {label} no JSON: {text[:200]}")
        return _fallback_compare(declared_codes, poste_info, "no JSON in response")

    # Normaliza campos obrigatórios
    parsed.setdefault("poste", poste_info)
    parsed.setdefault("estruturas_declaradas", declared_codes)
    parsed.setdefault("estruturas_confirmadas", [])
    parsed.setdefault("estruturas_divergentes", [])
    parsed.setdefault("materiais_visiveis", [])
    parsed.setdefault("materiais_faltantes", [])
    parsed.setdefault("conformidade", False)
    parsed.setdefault("observacoes", "")
    try:
        parsed["confianca"] = max(0.0, min(1.0, float(parsed.get("confianca", 0.0))))
    except (TypeError, ValueError):
        parsed["confianca"] = 0.0

    # Referências dos desenhos para o frontend renderizar
    parsed["norm_drawings"] = [
        {
            "codigo": code,
            "norm_id": norm_structs.get(code, {}).get("norm_id", ""),
            "pagina": norm_structs.get(code, {}).get("pagina_referencia", 0),
            "nome_completo": norm_structs.get(code, {}).get("nome_completo", code),
            "has_drawing": code in drawing_images,
        }
        for code in declared_codes
    ]

    logger.info(f"[vision-compare] {label} conformidade={parsed['conformidade']} "
                f"conf={parsed['confianca']:.2f} divergentes={parsed['estruturas_divergentes']}")
    return parsed


def _fallback_compare(declared_codes: list[str], poste_info: str, error: str) -> dict:
    return {
        "poste": poste_info,
        "estruturas_declaradas": declared_codes,
        "estruturas_confirmadas": [],
        "estruturas_divergentes": declared_codes,
        "materiais_visiveis": [],
        "materiais_faltantes": [],
        "conformidade": False,
        "confianca": 0.0,
        "observacoes": f"Erro na análise: {error[:200]}",
        "norm_drawings": [],
    }


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
