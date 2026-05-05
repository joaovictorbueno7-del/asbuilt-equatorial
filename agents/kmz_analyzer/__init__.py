from __future__ import annotations
import asyncio
from loguru import logger
from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register
from .parser import parse_kmz
from .vision import analyze_image

MAX_PARALLEL_VISION = 4


@register
class KMZAnalyzerAgent(BaseAgent):
    meta = AgentMeta(
        code="kmz_analyzer",
        name="KMZ Analyzer",
        description=(
            "Le arquivo KMZ, extrai fotos e coordenadas, analisa cada foto com "
            "Claude Vision, identifica estruturas eletricas e valida padroes tecnicos. "
            "Saida: JSON com estruturas + nao-conformidades + score de qualidade."
        ),
    )

    async def validate_input(self, payload):
        if not payload.get("kmz_path"):
            raise ValueError("payload.kmz_path is required")

    async def run(self, payload, *, context=None):
        kmz_path = payload["kmz_path"]
        examples = (context or {}).get("few_shot_examples", []) or []
        logger.info(f"[kmz_analyzer] start kmz={kmz_path} few_shot={len(examples)}")
        parsed = parse_kmz(kmz_path)
        placemarks = parsed["placemarks"]
        images = parsed["images"]
        logger.info(f"[kmz_analyzer] parsed: {len(placemarks)} placemarks, {len(images)} images")

        if not placemarks and not images:
            return AgentResult(
                output={"structures": [], "non_conformities": [], "quality_score": 0,
                        "image_count": 0, "placemark_count": 0,
                        "summary": "KMZ vazio: sem placemarks e sem imagens"},
                confidence=0.0, needs_human=True, notes="empty KMZ",
            )

        # Build (placemark_idx, image_key) pairs for analysis
        pairs: list[tuple[int | None, str]] = []
        for i, pm in enumerate(placemarks):
            for img_key in pm["images"]:
                pairs.append((i, img_key))

        # Fallback: if no placemark referenced any image, analyze all embedded images
        if not pairs and images:
            seen: set[bytes] = set()
            for img_key, data in images.items():
                fp = data[:64]
                if fp in seen:
                    continue
                seen.add(fp)
                pairs.append((None, img_key))

        sem = asyncio.Semaphore(MAX_PARALLEL_VISION)

        async def _bounded(img_key: str):
            async with sem:
                try:
                    return await analyze_image(images[img_key], image_label=img_key,
                                                examples=examples)
                except Exception as e:
                    logger.exception(f"[kmz_analyzer] image {img_key} failed: {e}")
                    return {
                        "structure_type": "outro", "condition": "regular",
                        "non_conformities": [], "confidence": 0.0,
                        "details": f"vision error: {type(e).__name__}: {e}",
                    }

        logger.info(f"[kmz_analyzer] analyzing {len(pairs)} images (max {MAX_PARALLEL_VISION} parallel)")
        analyses = await asyncio.gather(*[_bounded(k) for _, k in pairs]) if pairs else []
        logger.info(f"[kmz_analyzer] all images analyzed")

        structures: list[dict] = []
        non_conformities_global: list[dict] = []
        scores: list[float] = []
        learning_cases: list[dict] = []

        for (pm_idx, img_key), analysis in zip(pairs, analyses):
            entry: dict = {"image": img_key, "analysis": analysis}
            if pm_idx is not None:
                pm = placemarks[pm_idx]
                entry["placemark"] = {
                    "name": pm["name"], "lon": pm["lon"],
                    "lat": pm["lat"], "alt": pm["alt"],
                }
            structures.append(entry)
            for nc in analysis.get("non_conformities", []):
                non_conformities_global.append({
                    "image": img_key,
                    "issue": nc,
                    "placemark": placemarks[pm_idx]["name"] if pm_idx is not None else None,
                })
            try:
                scores.append(float(analysis.get("confidence", 0.0)))
            except (TypeError, ValueError):
                scores.append(0.0)
            learning_cases.append({
                "input": {"image_key": img_key,
                          "placemark": placemarks[pm_idx]["name"] if pm_idx is not None else None},
                "output": analysis,
            })

        avg_conf = sum(scores) / len(scores) if scores else 0.0
        # Quality 0-100: confidence baseline minus non-conformity penalty (capped)
        penalty = min(40, len(non_conformities_global) * 3)
        quality = max(0, min(100, int(avg_conf * 100 - penalty)))

        return AgentResult(
            output={
                "structures": structures,
                "non_conformities": non_conformities_global,
                "quality_score": quality,
                "image_count": len(structures),
                "placemark_count": len(placemarks),
                "summary": f"{len(structures)} fotos analisadas, "
                           f"{len(non_conformities_global)} nao-conformidades, "
                           f"qualidade {quality}/100",
            },
            confidence=avg_conf,
            needs_human=avg_conf < 0.70 or len(non_conformities_global) > 5,
            notes=f"analyzed {len(structures)} images across {len(placemarks)} placemarks",
            learning_cases=learning_cases,
        )
