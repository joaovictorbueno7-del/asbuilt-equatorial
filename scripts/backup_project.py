"""Backup completo do projeto: banco, normas e páginas renderizadas.

Uso:
    python scripts/backup_project.py
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = ROOT / "backups"


def run():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dest = BACKUP_ROOT / ts
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    errors: list[str] = []

    # 1. ops_ai_grid.db
    db = ROOT / "ops_ai_grid.db"
    if db.is_file():
        shutil.copy2(db, dest / "ops_ai_grid.db")
        copied.append(f"ops_ai_grid.db ({db.stat().st_size:,} bytes)")
    else:
        errors.append("ops_ai_grid.db não encontrado")

    # 2. knowledge/normas/
    normas_src = ROOT / "backend" / "knowledge" / "normas"
    if normas_src.is_dir():
        normas_dest = dest / "knowledge" / "normas"
        shutil.copytree(normas_src, normas_dest)
        pdfs = list(normas_dest.glob("*.pdf"))
        copied.append(f"knowledge/normas/ ({len(pdfs)} PDFs)")
    else:
        errors.append("backend/knowledge/normas/ não encontrado")

    # 3. knowledge/pages/ (se existir)
    pages_src = ROOT / "backend" / "knowledge" / "pages"
    if pages_src.is_dir():
        pages_dest = dest / "knowledge" / "pages"
        shutil.copytree(pages_src, pages_dest)
        imgs = list(pages_dest.rglob("*.jpg"))
        copied.append(f"knowledge/pages/ ({len(imgs)} imagens)")
    else:
        copied.append("knowledge/pages/ — não existe ainda (ok)")

    # 4. Git commit atual
    git_hash = "desconhecido"
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H %s"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception as e:
        errors.append(f"git log falhou: {e}")

    # 5. MANIFEST.txt
    manifest = dest / "MANIFEST.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"OPS AI GRID — Backup\n")
        f.write(f"Data/hora : {datetime.now().isoformat()}\n")
        f.write(f"Git commit: {git_hash}\n")
        f.write(f"Destino   : {dest}\n\n")
        f.write("Arquivos copiados:\n")
        for item in copied:
            f.write(f"  ✓ {item}\n")
        if errors:
            f.write("\nAvisos/erros:\n")
            for e in errors:
                f.write(f"  ✗ {e}\n")

    print(f"\n{'='*60}")
    print(f"Backup concluído em: backups/{ts}/")
    print(f"{'='*60}")
    for item in copied:
        print(f"  ✓ {item}")
    if errors:
        print("\nAvisos:")
        for e in errors:
            print(f"  ✗ {e}")
    print(f"\nMANIFEST: backups/{ts}/MANIFEST.txt")
    print(f"{'='*60}\n")
    return dest


if __name__ == "__main__":
    run()
