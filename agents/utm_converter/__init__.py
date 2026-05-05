import math
from agents.base import BaseAgent, AgentMeta, AgentResult
from agents.registry import register


def _utm_zone(lon: float) -> int:
    return int((lon + 180) / 6) + 1


def _wgs84_to_utm_naive(lat: float, lon: float) -> tuple[float, float, int, str]:
    """Naive UTM conversion (not survey-grade). Replaced by pyproj in real implementation."""
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = 2 * f - f * f
    zone = _utm_zone(lon)
    lon0 = math.radians(zone * 6 - 183)
    phi = math.radians(lat)
    lam = math.radians(lon)
    n = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = (e2 / (1 - e2)) * math.cos(phi) ** 2
    A = (lam - lon0) * math.cos(phi)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64) * phi
             - (3 * e2 / 8) * math.sin(2 * phi))
    x = k0 * n * (A + (1 - t + c) * A ** 3 / 6) + 500000.0
    y = k0 * (M + n * math.tan(phi) * (A ** 2 / 2 + (5 - t + 9 * c) * A ** 4 / 24))
    if lat < 0:
        y += 10000000.0
    hemi = "S" if lat < 0 else "N"
    return x, y, zone, hemi


@register
class UTMConverterAgent(BaseAgent):
    meta = AgentMeta(
        code="utm_converter",
        name="UTM Converter",
        description="Extrai coordenadas do KMZ e converte WGS84 -> UTM. "
                    "Exporta planilha. Valida area de concessao.",
    )

    async def run(self, payload, *, context=None):
        # Receives upstream from kmz_analyzer (parallel to 02-04, so reads 01 directly)
        structures = payload.get("structures", [])
        converted = []
        for s in structures:
            pm = s.get("placemark") or {}
            if "lat" in pm and "lon" in pm:
                x, y, zone, hemi = _wgs84_to_utm_naive(pm["lat"], pm["lon"])
                converted.append({
                    "name": pm.get("name", ""),
                    "wgs84": {"lat": pm["lat"], "lon": pm["lon"]},
                    "utm": {"easting": round(x, 2), "northing": round(y, 2),
                            "zone": zone, "hemisphere": hemi},
                    "policonico_pendente": True,
                })
        return AgentResult(
            output={
                "converted_points": converted,
                "out_of_concession": [],
                "excel_path": None,
                "policonico_conversion": "pendente: requer pyproj + grid SAD69/SIRGAS",
                "stub": True,
            },
            confidence=0.6,
            needs_human=False,
            notes=f"stub: {len(converted)} pontos convertidos para UTM (Policonico pendente)",
        )
