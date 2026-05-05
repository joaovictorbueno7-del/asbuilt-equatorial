"""Batch-index PDFs from a folder into the knowledge base via the HTTP API.

Usage:
    python scripts/index_normas_batch.py /path/to/folder --concessionaria Equatorial
    python scripts/index_normas_batch.py /path/to/folder -c Cemig --recursive --parallel 3

The script:
- walks the folder for *.pdf
- POSTs each one to /api/v1/knowledge in parallel (default 3 workers)
- detects duplicates server-side (HTTP 409 by MD5 hash) and counts them as skipped
- prints per-file progress and a final report

Auth: uses ADMIN_EMAIL / ADMIN_PASSWORD from .env.
"""
from __future__ import annotations
import argparse
import asyncio
import io
import re
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows console (cp1252 default chokes on unicode marks)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ENV = dotenv_values(str(ROOT / ".env"))

DEFAULT_BACKEND = ENV.get("BACKEND_URL", "http://127.0.0.1:8000")
ADMIN_EMAIL = ENV.get("ADMIN_EMAIL", "admin@opsaigrid.com")
ADMIN_PASSWORD = ENV.get("ADMIN_PASSWORD", "Admin@123")
API = "/api/v1"


def auto_codigo(filename: str) -> str:
    stem = Path(filename).stem
    m = re.match(r"^([A-Za-z]{1,5}[._-]\d+[._-][A-Za-z]+(?:[._-]\d+)?)", stem)
    if m:
        return m.group(1).upper().replace("_", ".")
    return stem[:40].upper()


def auto_nome(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"\s+", " ", stem.replace("_", " ").replace("-", " ").replace(".", " ")).strip()


async def login(client: httpx.AsyncClient) -> str:
    r = await client.post(f"{API}/auth/login",
                          json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    r.raise_for_status()
    return r.json()["access_token"]


async def upload_one(client: httpx.AsyncClient, token: str, pdf: Path,
                     concessionaria: str, versao: str) -> dict:
    codigo = auto_codigo(pdf.name)
    nome = auto_nome(pdf.name)
    with open(pdf, "rb") as f:
        files = {"file": (pdf.name, f.read(), "application/pdf")}
    data = {"concessionaria": concessionaria, "codigo": codigo, "nome": nome, "versao": versao}
    r = await client.post(
        f"{API}/knowledge",
        headers={"Authorization": f"Bearer {token}"},
        files=files, data=data, timeout=120.0,
    )
    if r.status_code == 409:
        return {"file": pdf.name, "status": "duplicate", "detail": r.json().get("detail", "")}
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text[:200])
        except Exception:
            detail = r.text[:200]
        return {"file": pdf.name, "status": "failed", "detail": detail}
    body = r.json()
    return {"file": pdf.name, "status": "uploaded", "norm_id": body["id"], "codigo": codigo}


async def wait_for_structures(client: httpx.AsyncClient, token: str, norm_id: str,
                                timeout: int = 90) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(3)
        r = await client.get(f"{API}/knowledge/{norm_id}",
                             headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            cnt = r.json().get("structure_count", 0)
            if cnt > 0:
                return cnt
    return 0


async def process_pdf(client: httpx.AsyncClient, token: str, sem: asyncio.Semaphore,
                      pdf: Path, concessionaria: str, versao: str) -> dict:
    async with sem:
        print(f"  [start ] {pdf.name}", flush=True)
        result = await upload_one(client, token, pdf, concessionaria, versao)
        if result["status"] == "uploaded":
            cnt = await wait_for_structures(client, token, result["norm_id"])
            result["structures"] = cnt
            result["status"] = "indexed"
            print(f"  [✓ done] {pdf.name}  ({cnt} estruturas)", flush=True)
        elif result["status"] == "duplicate":
            print(f"  [⊘ dup ] {pdf.name}", flush=True)
        else:
            print(f"  [✗ fail] {pdf.name}: {result.get('detail','')[:120]}", flush=True)
        return result


async def main_async(folder: Path, concessionaria: str, versao: str,
                     parallel: int, recursive: bool, backend: str):
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted([p for p in folder.glob(pattern) if p.is_file()])
    if not pdfs:
        print(f"Nenhum PDF encontrado em {folder} (recursive={recursive})")
        return 1

    print(f"Encontrados {len(pdfs)} PDFs. Concessionaria={concessionaria} | parallel={parallel}")
    print(f"Backend: {backend}\n")

    async with httpx.AsyncClient(base_url=backend) as client:
        try:
            token = await login(client)
        except httpx.HTTPError as e:
            print(f"Falha no login: {e}")
            return 2
        print(f"Login OK ({ADMIN_EMAIL})\n")

        sem = asyncio.Semaphore(parallel)
        tasks = [process_pdf(client, token, sem, p, concessionaria, versao) for p in pdfs]
        results = await asyncio.gather(*tasks)

    indexed = [r for r in results if r["status"] == "indexed"]
    duplicates = [r for r in results if r["status"] == "duplicate"]
    failed = [r for r in results if r["status"] == "failed"]
    total_structures = sum(r.get("structures", 0) for r in indexed)

    print("\n" + "=" * 60)
    print(f"RELATORIO FINAL")
    print("=" * 60)
    print(f"  Total processados   : {len(results)}")
    print(f"  Indexados           : {len(indexed)}  ({total_structures} estruturas)")
    print(f"  Duplicatas          : {len(duplicates)}")
    print(f"  Falhas              : {len(failed)}")

    if indexed:
        print("\n  ✓ INDEXADOS:")
        for r in indexed:
            print(f"      {r['codigo']:30s}  {r['structures']:3d} estruturas  {r['file']}")
    if duplicates:
        print("\n  ⊘ DUPLICATAS (já existiam, puladas):")
        for r in duplicates:
            print(f"      {r['file']}")
    if failed:
        print("\n  ✗ FALHAS:")
        for r in failed:
            print(f"      {r['file']}")
            print(f"          motivo: {r.get('detail','')[:200]}")

    return 0 if not failed else 3


def main():
    p = argparse.ArgumentParser(description="Batch-index norm PDFs into the knowledge base.")
    p.add_argument("folder", help="Pasta com os PDFs")
    p.add_argument("-c", "--concessionaria", required=True,
                   help="Concessionaria (Equatorial, Cemig, Copel, Enel, Light, EDP, Energisa, Outro)")
    p.add_argument("-v", "--versao", default="1.0", help="Versao (default 1.0)")
    p.add_argument("-p", "--parallel", type=int, default=3, help="PDFs em paralelo (default 3)")
    p.add_argument("-r", "--recursive", action="store_true", help="Buscar em subpastas")
    p.add_argument("--backend", default=DEFAULT_BACKEND, help=f"URL do backend (default {DEFAULT_BACKEND})")
    args = p.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Pasta nao existe: {folder}", file=sys.stderr)
        sys.exit(2)

    rc = asyncio.run(main_async(
        folder, args.concessionaria, args.versao, args.parallel, args.recursive, args.backend,
    ))
    sys.exit(rc)


if __name__ == "__main__":
    main()
