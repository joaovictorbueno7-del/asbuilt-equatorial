"""Build a synthetic KMZ for offline testing of the parser.
Two placemarks, each with one embedded JPEG (a small generated PIL image).
"""
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "test_sample.kmz"


def make_jpeg(text: str) -> bytes:
    img = Image.new("RGB", (640, 480), color=(40, 60, 90))
    d = ImageDraw.Draw(img)
    d.rectangle([200, 100, 440, 380], outline=(220, 220, 220), width=4)
    d.text((220, 220), text, fill=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Sample Network</name>
    <Placemark>
      <name>PT-001 Poste tipo I</name>
      <description><![CDATA[
        Poste de concreto na esquina.
        <img src="img/pt001.jpg" width="320"/>
      ]]></description>
      <Point><coordinates>-48.5044,-1.4558,12</coordinates></Point>
    </Placemark>
    <Placemark>
      <name>TR-014 Transformador 75kVA</name>
      <description><![CDATA[
        Transformador trifasico em poste duplo.
        <img src="img/tr014.jpg"/>
      ]]></description>
      <Point><coordinates>-48.5051,-1.4571,11</coordinates></Point>
    </Placemark>
  </Document>
</kml>
"""


def main():
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", KML)
        zf.writestr("img/pt001.jpg", make_jpeg("PT-001"))
        zf.writestr("img/tr014.jpg", make_jpeg("TR-014"))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
