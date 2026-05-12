"""Migração Opção A: adiciona colunas faltantes sem perder dados existentes.

Colunas adicionadas:
  technical_norms:
    - processing_status, processing_progress, processing_message
    - processing_started_at, processing_finished_at
    - pages_total, pages_with_drawings, pages_processed

  norm_structures:
    - tipo_rede, tensao_nominal, como_identificar_na_foto
    - restricoes_uso, desenho_numero, pagina_referencia
    - imagem_desenho_path, fixacao (JSON), source_text_excerpt
    - extraction_confidence, requires_review, parent_structure_id

Tabela criada:
  norm_materials (completa)

Uso:
    python scripts/migrate_db.py
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "ops_ai_grid.db"


def col_exists(cursor, table: str, col: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == col for row in cursor.fetchall())


def table_exists(cursor, table: str) -> bool:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cursor.fetchone() is not None


def run():
    if not DB_PATH.is_file():
        print(f"ERRO: banco nao encontrado em {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    added: list[str] = []
    skipped: list[str] = []

    # ─── technical_norms ────────────────────────────────────────────────────────
    tn_cols = [
        ("processing_status",    "VARCHAR(20) NOT NULL DEFAULT 'idle'"),
        ("processing_progress",  "INTEGER NOT NULL DEFAULT 0"),
        ("processing_message",   "VARCHAR(500) NOT NULL DEFAULT ''"),
        ("processing_started_at","DATETIME"),
        ("processing_finished_at","DATETIME"),
        ("pages_total",          "INTEGER NOT NULL DEFAULT 0"),
        ("pages_with_drawings",  "INTEGER NOT NULL DEFAULT 0"),
        ("pages_processed",      "INTEGER NOT NULL DEFAULT 0"),
    ]
    for col, typedef in tn_cols:
        if col_exists(c, "technical_norms", col):
            skipped.append(f"technical_norms.{col}")
        else:
            c.execute(f"ALTER TABLE technical_norms ADD COLUMN {col} {typedef}")
            added.append(f"technical_norms.{col}")

    # ─── norm_structures ────────────────────────────────────────────────────────
    ns_cols = [
        ("tipo_rede",               "VARCHAR(20) NOT NULL DEFAULT ''"),
        ("tensao_nominal",          "VARCHAR(40) NOT NULL DEFAULT ''"),
        ("como_identificar_na_foto","TEXT NOT NULL DEFAULT ''"),
        ("restricoes_uso",          "TEXT NOT NULL DEFAULT ''"),
        ("desenho_numero",          "VARCHAR(40) NOT NULL DEFAULT ''"),
        ("pagina_referencia",       "INTEGER NOT NULL DEFAULT 0"),
        ("imagem_desenho_path",     "VARCHAR(1000) NOT NULL DEFAULT ''"),
        ("fixacao",                 "JSON NOT NULL DEFAULT '{}'"),
        ("source_text_excerpt",     "TEXT NOT NULL DEFAULT ''"),
        ("extraction_confidence",   "REAL NOT NULL DEFAULT 0.0"),
        ("requires_review",         "BOOLEAN NOT NULL DEFAULT 0"),
        ("parent_structure_id",     "VARCHAR(36)"),
    ]
    for col, typedef in ns_cols:
        if col_exists(c, "norm_structures", col):
            skipped.append(f"norm_structures.{col}")
        else:
            c.execute(f"ALTER TABLE norm_structures ADD COLUMN {col} {typedef}")
            added.append(f"norm_structures.{col}")

    # ─── norm_materials (CREATE TABLE se nao existir) ───────────────────────────
    if table_exists(c, "norm_materials"):
        skipped.append("tabela norm_materials (ja existe)")
    else:
        c.execute("""
            CREATE TABLE norm_materials (
                id                VARCHAR(36)  NOT NULL PRIMARY KEY,
                created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                tenant_id         VARCHAR(36)  NOT NULL,
                norm_id           VARCHAR(36)  NOT NULL,
                codigo_material   VARCHAR(80)  NOT NULL,
                codigo_item       VARCHAR(40)  NOT NULL DEFAULT '',
                descricao         TEXT         NOT NULL DEFAULT '',
                tensao            VARCHAR(40)  NOT NULL DEFAULT '',
                used_in_structures JSON        NOT NULL DEFAULT '[]',
                FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                FOREIGN KEY (norm_id)   REFERENCES technical_norms(id)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS ix_norm_materials_tenant_id  ON norm_materials(tenant_id)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_norm_materials_norm_id    ON norm_materials(norm_id)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_norm_materials_codigo_mat ON norm_materials(codigo_material)")
        added.append("tabela norm_materials (criada com indices)")

    conn.commit()
    conn.close()

    print("\n" + "="*60)
    print("MIGRACAO CONCLUIDA")
    print("="*60)
    if added:
        print(f"\nAdicionado ({len(added)}):")
        for item in added:
            print(f"  + {item}")
    if skipped:
        print(f"\nJa existia ({len(skipped)}) — nao alterado:")
        for item in skipped:
            print(f"  = {item}")
    print(f"\nBanco: {DB_PATH}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run()
