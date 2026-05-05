"""Parse KMZ archives: extract placemarks (name, coords, description) and embedded images.

KMZ = ZIP containing one .kml file (usually doc.kml) plus optional media (jpg/png).
Placemarks may reference images via <img src="..."> or <a href="..."> in their HTML
description, or they may be orphan (no association). We return both: placemarks with
matched images, and the full image map so the caller can fall back to analyzing
unmatched images.
"""
from __future__ import annotations
import zipfile
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict


class Placemark(TypedDict):
    name: str
    description: str
    lon: float
    lat: float
    alt: float
    images: list[str]


class ParsedKMZ(TypedDict):
    placemarks: list[Placemark]
    images: dict[str, bytes]


_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _localname(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_kmz(kmz_path: str) -> ParsedKMZ:
    p = Path(kmz_path)
    if not p.is_file():
        raise FileNotFoundError(kmz_path)

    with zipfile.ZipFile(p, "r") as zf:
        names = zf.namelist()
        kml_name = next((n for n in names if n.lower().endswith("doc.kml")), None)
        if not kml_name:
            kml_name = next((n for n in names if n.lower().endswith(".kml")), None)
        if not kml_name:
            raise ValueError("KMZ does not contain any .kml file")
        kml_bytes = zf.read(kml_name)

        images: dict[str, bytes] = {}
        for n in names:
            ext = Path(n).suffix.lower()
            if ext in _IMG_EXT:
                data = zf.read(n)
                images[n] = data
                base = Path(n).name
                if base != n and base not in images:
                    images[base] = data

    try:
        root = ET.fromstring(kml_bytes)
    except ET.ParseError as e:
        raise ValueError(f"Invalid KML XML: {e}")

    placemarks: list[Placemark] = []
    for pm in root.iter():
        if _localname(pm.tag) != "Placemark":
            continue
        name = ""
        desc = ""
        coords_text = ""
        for child in pm.iter():
            ln = _localname(child.tag)
            if ln == "name" and child.text and not name:
                name = child.text.strip()
            elif ln == "description" and child.text and not desc:
                desc = child.text.strip()
            elif ln == "coordinates" and child.text and not coords_text:
                coords_text = child.text.strip().split()[0]

        if not coords_text:
            continue
        parts = coords_text.split(",")
        try:
            lon_f = float(parts[0])
            lat_f = float(parts[1])
            alt_f = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        except (ValueError, IndexError):
            continue

        img_refs: list[str] = []
        if desc:
            img_refs += re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', desc, re.I)
            img_refs += re.findall(
                r'href=["\']([^"\']+\.(?:jpg|jpeg|png|webp))["\']', desc, re.I
            )

        matched: list[str] = []
        for ref in img_refs:
            ref_clean = ref.strip().lstrip("./").split("#", 1)[0].split("?", 1)[0]
            if ref_clean in images:
                matched.append(ref_clean)
            elif Path(ref_clean).name in images:
                matched.append(Path(ref_clean).name)

        placemarks.append(Placemark(
            name=name, description=desc[:1000],
            lon=lon_f, lat=lat_f, alt=alt_f, images=matched,
        ))

    return ParsedKMZ(placemarks=placemarks, images=images)
