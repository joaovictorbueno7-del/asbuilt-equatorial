"""KMZ Analyzer Agent — Agente 01.

Fluxo real:
  1. Parse KMZ → placemarks + imagens
  2. Para cada placemark: extrai códigos declarados (ex: "SI3 N1")
  3. Busca estruturas na base de normas (NT.00022)
  4. Chama Claude Vision com:
       - Foto do campo
       - Desenhos técnicos das estruturas declaradas
       - Lista de materiais esperados
  5. Retorna análise com conformidade, divergências e casos de aprendizado
"""
from __future__ import annotations
import asyncio
from loguru import logger
from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register
from .parser import parse_kmz
from .vision import analyze_image, compare_with_norm

MAX_PARALLEL_VISION = 3  # conservador para rate limit
NT00022_ID = "d06b92ba-37c9-4c45-8bf1-e41633674559"


def _try_import_norm_lookup():
    """Import com fallback gracioso caso backend não esteja no path."""
    try:
        from app.services.norm_lookup import enrich_with_norm
        return enrich_with_norm
    except ImportError:
        logger.warning("[kmz_analyzer] norm_lookup não disponível — sem comparação com norma")
        return None


@register
class KMZAnalyzerAgent(BaseAgent):
    meta = AgentMeta(
        code="kmz_analyzer",
        name="KMZ Analyzer",
        description=(
            "Le arquivo KMZ, extrai fotos e coordenadas, identifica estruturas "
            "declaradas (ex: N1, S3I), busca desenho técnico na base de normas "
            "(NT.00022), compara foto do campo com a norma via Claude Vision e "
            "detecta divergências. Saída: conformidade, divergências, materiais "
            "faltantes e score de qualidade."
        ),
    )

    async def validate_input(self, payload):
        if not payload.get("kmz_path"):
            raise ValueError("payload.kmz_path is required")

    async def run(self, payload, *, context=None):
        kmz_path = payload["kmz_path"]
        examples = (context or {}).get("few_shot_examples", []) or []
        enrich_with_norm = _try_import_norm_lookup()

        logger.info(f"[kmz_analyzer] start kmz={kmz_path} few_shot={len(examples)} "
                    f"norm_lookup={'sim' if enrich_with_norm else 'nao'}")

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

        # Constrói pares (placemark_idx, image_key)
        pairs: list[tuple[int | None, str]] = []
        for i, pm in enumerate(placemarks):
            for img_key in pm["images"]:
                pairs.append((i, img_key))

        # Fallback: sem referências → analisa todas as imagens
        if not pairs and images:
            seen: set[bytes] = set()
            for img_key, data in images.items():
                fp = data[:64]
                if fp in seen:
                    continue
                seen.add(fp)
                pairs.append((None, img_key))

        sem = asyncio.Semaphore(MAX_PARALLEL_VISION)

        async def _analyze_one(pm_idx: int | None, img_key: str) -> dict:
            """Analisa uma imagem — com comparação normativa se disponível."""
            async with sem:
                raw_photo = images.get(img_key, b"")
                if not raw_photo:
                    return _empty_result(img_key, "imagem não encontrada no KMZ")

                pm = placemarks[pm_idx] if pm_idx is not None else None
                pm_name = pm["name"] if pm else ""
                pm_desc = pm.get("description", "") if pm else ""

                # Tenta comparação com norma
                if enrich_with_norm and (pm_name or pm_desc):
                    try:
                        norm_data = enrich_with_norm(
                            pm_name, pm_desc, norm_id=NT00022_ID
                        )
                        found = norm_data["found"]
                        declared = norm_data["declared_codes"]
                        drawings = norm_data["drawings"]

                        if found:
                            logger.info(
                                f"[kmz_analyzer] {img_key}: declarados={declared} "
                                f"encontrados={list(found.keys())}"
                            )
                            analysis = await compare_with_norm(
                                raw_photo,
                                declared_codes=list(found.keys()),
                                norm_structs=found,
                                drawing_images=drawings,
                                poste_info=pm_name,
                                image_label=img_key,
                            )
                            # Preenche campos legacy para compatibilidade com pipeline
                            analysis = _enrich_legacy_fields(analysis, pm_name)
                            return analysis
                        else:
                            logger.debug(
                                f"[kmz_analyzer] {img_key}: nenhuma estrutura "
                                f"encontrada na norma para {declared}"
                            )
                    except Exception as e:
                        logger.warning(f"[kmz_analyzer] norm compare falhou {img_key}: {e}")

                # Fallback: análise básica sem norma
                try:
                    return await analyze_image(
                        raw_photo, image_label=img_key, examples=examples
                    )
                except Exception as e:
                    logger.exception(f"[kmz_analyzer] {img_key} vision failed: {e}")
                    return _empty_result(img_key, f"vision error: {e}")

        logger.info(f"[kmz_analyzer] analisando {len(pairs)} imagens "
                    f"(max {MAX_PARALLEL_VISION} paralelas)")
        analyses = await asyncio.gather(*[_analyze_one(pm_idx, k) for pm_idx, k in pairs])
        logger.info(f"[kmz_analyzer] todas as imagens analisadas")

        # Agrega resultados
        structures: list[dict] = []
        non_conformities_global: list[dict] = []
        scores: list[float] = []
        learning_cases: list[dict] = []
        total_divergencias = 0

        for (pm_idx, img_key), analysis in zip(pairs, analyses):
            pm = placemarks[pm_idx] if pm_idx is not None else None
            entry: dict = {
                "image": img_key,
                "analysis": analysis,
            }
            if pm:
                entry["placemark"] = {
                    "name": pm["name"],
                    "lon": pm["lon"],
                    "lat": pm["lat"],
                    "alt": pm["alt"],
                }

            structures.append(entry)

            # Non-conformidades (legado + divergências normativas)
            for nc in analysis.get("non_conformities", []):
                non_conformities_global.append({
                    "image": img_key,
                    "issue": nc,
                    "placemark": pm["name"] if pm else None,
                })
            for div in analysis.get("estruturas_divergentes", []):
                non_conformities_global.append({
                    "image": img_key,
                    "issue": f"Estrutura divergente: {div}",
                    "tipo": "divergencia_normativa",
                    "placemark": pm["name"] if pm else None,
                })
            for mat in analysis.get("materiais_faltantes", []):
                non_conformities_global.append({
                    "image": img_key,
                    "issue": f"Material faltante: {mat}",
                    "tipo": "material_faltante",
                    "placemark": pm["name"] if pm else None,
                })

            if not analysis.get("conformidade", True):
                total_divergencias += 1

            # Confiança
            conf = analysis.get("confianca") or analysis.get("confidence") or 0.0
            try:
                scores.append(float(conf))
            except (TypeError, ValueError):
                scores.append(0.0)

            # Caso de aprendizado
            learning_cases.append({
                "input": {
                    "image_key": img_key,
                    "placemark": pm["name"] if pm else None,
                    "estruturas_declaradas": analysis.get("estruturas_declaradas", []),
                },
                "output": analysis,
                "norm_drawings": analysis.get("norm_drawings", []),
                "conformidade": analysis.get("conformidade"),
            })

        avg_conf = sum(scores) / len(scores) if scores else 0.0
        penalty = min(50, len(non_conformities_global) * 3 + total_divergencias * 10)
        quality = max(0, min(100, int(avg_conf * 100 - penalty)))

        total_conformes = sum(
            1 for s in structures
            if s["analysis"].get("conformidade") is True
        )
        total_com_norma = sum(
            1 for s in structures
            if s["analysis"].get("estruturas_declaradas")
        )

        return AgentResult(
            output={
                "structures": structures,
                "non_conformities": non_conformities_global,
                "quality_score": quality,
                "image_count": len(structures),
                "placemark_count": len(placemarks),
                "total_conformes": total_conformes,
                "total_divergentes": total_divergencias,
                "total_com_norma": total_com_norma,
                "summary": (
                    f"{len(structures)} fotos analisadas | "
                    f"{total_conformes} conformes | "
                    f"{total_divergencias} divergências | "
                    f"qualidade {quality}/100"
                ),
            },
            confidence=avg_conf,
            needs_human=(
                avg_conf < 0.70
                or len(non_conformities_global) > 5
                or total_divergencias > 0
            ),
            notes=(
                f"analyzed {len(structures)} images across {len(placemarks)} placemarks; "
                f"{total_com_norma} with norm comparison"
            ),
            learning_cases=learning_cases,
        )


def _enrich_legacy_fields(norm_analysis: dict, pm_name: str) -> dict:
    """Adiciona campos legacy (structure_type, condition, confidence) para
    compatibilidade com agentes downstream que leem esses campos."""
    if "structure_type" not in norm_analysis:
        # Infere tipo pelo que foi confirmado
        confirmed = norm_analysis.get("estruturas_confirmadas", [])
        if confirmed:
            code = confirmed[0].upper()
            if any(t in code for t in ["TR", "TRAFO", "BS", "NS"]):
                norm_analysis["structure_type"] = "transformador"
            elif any(t in code for t in ["CHAVE", "CF", "PR-N"]):
                norm_analysis["structure_type"] = "chave"
            elif any(t in code for t in ["MUFLA"]):
                norm_analysis["structure_type"] = "isolador"
            else:
                norm_analysis["structure_type"] = "poste"
        else:
            norm_analysis["structure_type"] = "poste"

    if "condition" not in norm_analysis:
        if norm_analysis.get("conformidade"):
            norm_analysis["condition"] = "boa"
        elif norm_analysis.get("estruturas_divergentes"):
            norm_analysis["condition"] = "ruim"
        else:
            norm_analysis["condition"] = "regular"

    if "confidence" not in norm_analysis:
        norm_analysis["confidence"] = norm_analysis.get("confianca", 0.0)

    if "non_conformities" not in norm_analysis:
        ncs = list(norm_analysis.get("estruturas_divergentes", []))
        ncs += [f"material faltante: {m}" for m in norm_analysis.get("materiais_faltantes", [])]
        norm_analysis["non_conformities"] = ncs

    if "details" not in norm_analysis:
        obs = norm_analysis.get("observacoes", "")
        confirmados = norm_analysis.get("estruturas_confirmadas", [])
        norm_analysis["details"] = (
            f"Confirmadas: {', '.join(confirmados) or 'nenhuma'}. {obs}"
        )[:300]

    return norm_analysis


def _empty_result(img_key: str, reason: str) -> dict:
    return {
        "structure_type": "outro",
        "condition": "regular",
        "non_conformities": [],
        "confidence": 0.0,
        "confianca": 0.0,
        "details": reason,
        "conformidade": None,
        "estruturas_declaradas": [],
        "estruturas_confirmadas": [],
        "estruturas_divergentes": [],
        "materiais_visiveis": [],
        "materiais_faltantes": [],
        "observacoes": reason,
        "norm_drawings": [],
    }
