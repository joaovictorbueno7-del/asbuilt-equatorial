"""Daily backup of the knowledge base + SQLite DB.

Layout produced:
  backups/
    YYYY-MM-DD/
      ops_ai_grid.db
      knowledge/normas/<all PDFs>
      backup.log

Retention: keeps last 30 daily folders. Older are removed automatically.

Schedule on Windows (run once in PowerShell as admin):
  $action  = New-ScheduledTaskAction -Execute "C:\\Users\\FILIPE-PSE\\.claude\\agentes de medicao\\.venv\\Scripts\\python.exe" -Argument "C:\\Users\\FILIPE-PSE\\.claude\\agentes de medicao\\scripts\\backup_knowledge.py"
  $trigger = New-ScheduledTaskTrigger -Daily -At 2am
  Register-ScheduledTask -TaskName "OpsAiGrid-DailyBackup" -Action $action -Trigger $trigger -RunLevel Highest

Manual run: python scripts/backup_knowledge.py
"""
from __future__ import annotations
import shutil
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP_ROOT = ROOT / "backups"
RETENTION_DAYS = 30

DB_FILE = ROOT / "ops_ai_grid.db"
KNOWLEDGE_DIR = ROOT / "backend" / "knowledge"

# pypdf etc not needed; pure stdlib


def main():
    today = date.today().isoformat()
    target = BACKUP_ROOT / today
    target.mkdir(parents=True, exist_ok=True)

    log_lines = [f"[{datetime.now().isoformat()}] backup start -> {target}"]

    # 1. DB
    if DB_FILE.is_file():
        dst = target / DB_FILE.name
        shutil.copy2(DB_FILE, dst)
        log_lines.append(f"DB: copied {DB_FILE.stat().st_size} bytes -> {dst.name}")
    else:
        log_lines.append(f"DB: MISSING ({DB_FILE})")

    # 2. knowledge/ tree
    if KNOWLEDGE_DIR.is_dir():
        dst_dir = target / "knowledge"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(KNOWLEDGE_DIR, dst_dir)
        n_files = sum(1 for _ in dst_dir.rglob("*") if _.is_file())
        total_bytes = sum(p.stat().st_size for p in dst_dir.rglob("*") if p.is_file())
        log_lines.append(f"knowledge: {n_files} files, {total_bytes} bytes")
    else:
        log_lines.append(f"knowledge: dir not found ({KNOWLEDGE_DIR})")

    # 3. retention
    cutoff = date.today() - timedelta(days=RETENTION_DAYS)
    removed = 0
    for entry in BACKUP_ROOT.iterdir() if BACKUP_ROOT.is_dir() else []:
        if not entry.is_dir():
            continue
        try:
            d = date.fromisoformat(entry.name)
        except ValueError:
            continue
        if d < cutoff:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    log_lines.append(f"retention: removed {removed} folders older than {RETENTION_DAYS} days")

    log_lines.append(f"[{datetime.now().isoformat()}] backup done")
    log_path = target / "backup.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    # also append to global log
    global_log = BACKUP_ROOT / "backup_history.log"
    global_log.parent.mkdir(parents=True, exist_ok=True)
    with open(global_log, "a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")
        f.write("---\n")

    for line in log_lines:
        print(line)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"BACKUP FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
