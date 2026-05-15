"""Supervised learning cases: upload real field photos with correct labels."""
from __future__ import annotations
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.models import User, LearningCase
from app.auth.dependencies import get_current_user

router = APIRouter()

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


class CaseOut(BaseModel):
    id: str
    structure_codes: list[str]
    pole_size: str
    conformidade: bool
    notes: str
    concessionaria: str
    created_at: datetime


def _resize_image(data: bytes, max_px: int = 1200) -> bytes:
    """Resize image to max_px x max_px keeping aspect ratio, return JPEG bytes."""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        # If PIL not available or image cannot be processed, return original
        return data


@router.post("", response_model=CaseOut, status_code=201)
async def create_case(
    file: UploadFile = File(...),
    structure_codes: str = Form(...),
    pole_size: str = Form(default=""),
    conformidade: str = Form(...),
    notes: str = Form(default=""),
    concessionaria: str = Form(default=""),
    bbox: str = Form(default=""),   # JSON string: {"x":0.1,"y":0.2,"w":0.4,"h":0.3}
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "File must be jpg or png")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB")

    # Resize image
    content = _resize_image(content)

    # Save to storage
    storage = Path(settings.STORAGE_LOCAL_PATH) / "learning"
    storage.mkdir(parents=True, exist_ok=True)
    img_path = storage / f"{uuid.uuid4()}.jpg"
    img_path.write_bytes(content)

    # Parse fields
    codes = [c.strip() for c in structure_codes.split(",") if c.strip()]
    is_conforme = conformidade.lower() in ("true", "1", "yes", "sim")

    # Parse optional bbox annotation
    bbox_data: dict | None = None
    if bbox.strip():
        import json as _json
        try:
            parsed = _json.loads(bbox)
            if isinstance(parsed, dict) and all(k in parsed for k in ("x", "y", "w", "h")):
                bbox_data = {k: max(0.0, min(1.0, float(parsed[k]))) for k in ("x", "y", "w", "h")}
        except Exception:
            pass

    case = LearningCase(
        tenant_id=user.tenant_id,
        agent_code="kmz_analyzer",
        source_run_id=None,
        is_correct=True,
        times_used=1,
        feedback_score=1.0,
        input_payload={
            "image_path": str(img_path),
            "structure_codes": codes,
            "pole_size": pole_size,
            "concessionaria": concessionaria,
            **({"bbox": bbox_data} if bbox_data else {}),
        },
        expected_output={
            "conformidade": is_conforme,
            "estruturas_confirmadas": codes,
            "structure_type": "poste",
            "confianca": 1.0,
            "observacoes": notes,
        },
        observed_output={},
        human_notes=notes,
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)

    return CaseOut(
        id=case.id,
        structure_codes=codes,
        pole_size=pole_size,
        conformidade=is_conforme,
        notes=notes,
        concessionaria=concessionaria,
        created_at=case.created_at,
    )


@router.get("", response_model=list[dict])
async def list_cases(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    rows = (await db.execute(
        select(LearningCase)
        .where(
            LearningCase.tenant_id == user.tenant_id,
            LearningCase.agent_code == "kmz_analyzer",
        )
        .order_by(desc(LearningCase.created_at))
        .limit(min(limit, 200))
    )).scalars().all()

    result = []
    for r in rows:
        inp = r.input_payload or {}
        exp = r.expected_output or {}
        result.append({
            "id": r.id,
            "structure_codes": inp.get("structure_codes", []),
            "pole_size": inp.get("pole_size", ""),
            "concessionaria": inp.get("concessionaria", ""),
            "conformidade": exp.get("conformidade", True),
            "notes": r.human_notes or "",
            "created_at": r.created_at.isoformat(),
        })
    return result


@router.get("/{case_id}/image")
async def get_case_image(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    case = (await db.execute(
        select(LearningCase).where(
            LearningCase.id == case_id,
            LearningCase.tenant_id == user.tenant_id,
        )
    )).scalar_one_or_none()
    if not case:
        raise HTTPException(404, "Case not found")

    img_path = Path(case.input_payload.get("image_path", ""))
    if not img_path.is_file():
        raise HTTPException(404, "Image file missing")

    data = img_path.read_bytes()
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.delete("/{case_id}", status_code=204)
async def delete_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    case = (await db.execute(
        select(LearningCase).where(
            LearningCase.id == case_id,
            LearningCase.tenant_id == user.tenant_id,
        )
    )).scalar_one_or_none()
    if not case:
        raise HTTPException(404, "Case not found")

    # Delete image file
    img_path = Path(case.input_payload.get("image_path", ""))
    if img_path.is_file():
        try:
            img_path.unlink()
        except OSError:
            pass

    await db.delete(case)
    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Detecção visual: Claude Vision localiza o elemento na foto
# ─────────────────────────────────────────────────────────────────────────────

DETECT_PROMPT = """Você é um especialista em redes elétricas de distribuição.
Analise esta foto de campo e localize o elemento indicado.

Elemento a localizar: {label}

Retorne APENAS um JSON válido (nada antes ou depois):
{{
  "found": true,
  "bbox": {{"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3}},
  "confidence": 0.85,
  "description": "descrição curta do que foi encontrado (max 100 chars)"
}}

Regras:
- bbox usa coordenadas RELATIVAS (0.0 a 1.0) em relação ao tamanho total da imagem
  - x, y = canto superior esquerdo da caixa
  - w, h = largura e altura da caixa
- Se não encontrar o elemento: {{"found": false, "bbox": null, "confidence": 0, "description": "não encontrado"}}
- confidence entre 0.0 e 1.0
- Para postes: localize a placa amarela com número ou a marcação estampada no concreto
- Para estruturas MT/BT: localize os ferros, cruzetas, transformadores ou isoladores indicados
"""


@router.post("/detect")
async def detect_structure(
    file: UploadFile = File(...),
    label: str = Form(...),   # ex: "MT:N1", "BT:R3", "POSTE:12/600 nº47022814"
    user: User = Depends(get_current_user),
):
    """Chama Claude Vision para localizar o elemento na foto e retorna bbox."""
    import base64, json, re
    from anthropic import AsyncAnthropic
    from app.core.config import settings

    content = await file.read()
    if not content:
        raise HTTPException(400, "Arquivo vazio")

    # Normaliza imagem
    img_bytes = _resize_image(content, max_px=1200)

    b64 = base64.standard_b64encode(img_bytes).decode()
    prompt = DETECT_PROMPT.format(label=label)

    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY não configurada")

    client = AsyncAnthropic(api_key=api_key)
    try:
        msg = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
    except Exception as e:
        raise HTTPException(500, f"Erro Vision: {e}")

    text = "".join(getattr(b, "text", "") for b in msg.content)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"found": False, "bbox": None, "confidence": 0, "description": "Sem resposta"}

    try:
        result = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"found": False, "bbox": None, "confidence": 0, "description": "JSON inválido"}

    # Garante bbox válido (valores entre 0 e 1)
    bbox = result.get("bbox")
    if bbox and isinstance(bbox, dict):
        for k in ("x", "y", "w", "h"):
            bbox[k] = max(0.0, min(1.0, float(bbox.get(k, 0))))

    return {
        "found": result.get("found", False),
        "bbox": bbox,
        "confidence": float(result.get("confidence", 0)),
        "description": str(result.get("description", ""))[:200],
    }
