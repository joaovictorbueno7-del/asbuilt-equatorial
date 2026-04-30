# Arquitetura — OPS AI GRID

## Princípios

- **Clean architecture**: domain (models) → services → API. Sem lógica em routers.
- **Plugin-based**: agentes registram-se via decorator. Adicionar agente não toca core.
- **Abstração de infra**: DB, fila e storage atrás de interfaces — troca por env var.
- **Multi-tenant by default**: toda query escopada de tenant deve filtrar `tenant_id`.
- **Audit-first**: ações sensíveis (auth, agent runs, mudanças de role) gravam `AuditLog`.

## Camadas

```
┌──────────────────────────────────────────────────┐
│  API (FastAPI routers — backend/app/api/v1)      │
├──────────────────────────────────────────────────┤
│  Auth + RBAC (dependencies, MFA, JWT)            │
├──────────────────────────────────────────────────┤
│  Services (audit, learning, pipeline)            │
├──────────────────────────────────────────────────┤
│  Agents (plugin registry — agents/)              │
├──────────────────────────────────────────────────┤
│  Core (config, db, security, queue, logging)     │
├──────────────────────────────────────────────────┤
│  Storage: SQLite/Postgres + (local/S3) + Redis?  │
└──────────────────────────────────────────────────┘
```

## Pipeline padrão (caso completo de as-built)

```
KMZ upload
   │
   ▼
[01 kmz_analyzer]──► [02 description_filler]──► [03 report_generator]──► [04 anti_reprova]
   │                                                                            │
   └──────────────────[05 pipeline_supervisor monitora 01–04]───────────────────┘

[06 utm_converter] roda em paralelo sobre o KMZ
[07 rpa_screen_learner] preenche PROJ via Citrix (computer-use)
[08 adherence_tester] cruza material × serviço × valor
[09 master_supervisor] valida tudo, gera score 0–100, libera envio
```

Cada agente grava um `AgentRun`. Pipelines encadeiam runs via `parent_run_id`.

## Aprendizado (LearningCase)

Cada agente acumula `LearningCase` com input + output esperado + output observado +
feedback humano. Usados para:

- **Shadow mode** do RPA Screen Learner (10 exemplos antes de assumir)
- Calibração de confidence_score
- Detecção de drift (queda no acerto histórico)
- Few-shot prompting do Claude com casos parecidos

## Decisões em aberto (TODO)

- Substituir slowapi (in-memory) por Redis quando Celery entrar
- Frontend Next.js (próxima sprint)
- Implementar S3/MinIO storage backend
- Alembic para migrações (atualmente `create_all`)
- Webhook de eventos para integradores externos
