# Políticas de Segurança — OPS AI GRID

## Senhas
- Mínimo 10 caracteres no registro (validado em Pydantic)
- Hash Argon2id (time_cost=3, memory=64MiB, parallelism=2)
- Rehash automático quando parâmetros mudam
- Lockout: 5 tentativas falhas → 15min bloqueado

## Tokens
- Access JWT: 15min, contém `sub`, `tenant_id`, `role`, `email`, `jti`
- Refresh JWT: 7d, `jti` salvo em `refresh_tokens` para revogação
- Rotação: `/auth/refresh` revoga o token anterior e emite par novo
- Logout: marca `revoked=true` no refresh

## MFA
- TOTP via pyotp (RFC 6238), janela ±30s
- Secret e códigos de recuperação criptografados com Fernet (chave `FERNET_KEY` no env)
- Setup → enable em duas etapas (evita bloquear conta com QR não escaneado)

## Multi-tenant
- Todas tabelas escopadas têm `tenant_id` (FK CASCADE)
- Toda query de domínio DEVE filtrar `tenant_id` do usuário autenticado
- JWT carrega `tenant_id` para evitar lookup extra

## RBAC
- Papéis: `admin`, `supervisor`, `operador`, `auditor`
- `admin` tem acesso total dentro do próprio tenant
- Decorator `require_roles(...)` em endpoints sensíveis
- Mudanças de role gravam audit log

## Audit log
- Append-only: aplicação nunca emite UPDATE/DELETE em `audit_logs`
- Eventos mínimos: login, login_failed, mfa_enabled/disabled/failed, role_changed,
  agent_run_created, sensitive_export
- Em produção: replicar para storage WORM (S3 Object Lock)

## Defesas OWASP top 10
- **Injection**: SQLAlchemy bound params, jamais string interp
- **Broken auth**: JWT + MFA + lockout + rotação
- **Sensitive data**: Fernet em colunas sensíveis, TLS obrigatório em prod
- **XXE**: não parsear XML de input não confiável (KMZ usa zipfile + xml seguro)
- **Broken access control**: tenant_id check em toda query
- **Security misconfig**: headers via middleware, CORS por allowlist
- **XSS**: API JSON-only, sem render server-side
- **Deserialization**: Pydantic com modelos estritos
- **Components com vuln**: pin de versões em requirements.txt, dependabot
- **Logging insufficient**: AuditLog + loguru estruturado

## Upload de arquivo (futuro)
- Scan antivírus (ClamAV) antes de persistir
- Limite de tamanho via FastAPI
- Sandbox de extração de KMZ (zipfile.ZipFile + path traversal check)
- Storage fora do diretório web servido

## Backup
- SQLite: copiar arquivo (WAL mode permite hot copy)
- Postgres: pg_dump diário + WAL archiving
- Retenção: 30d quente, 1 ano frio

## Resposta a incidentes
- Revogação imediata: `UPDATE refresh_tokens SET revoked=true WHERE user_id=?`
- Forçar logout global de tenant: rotacionar `SECRET_KEY` (invalida todos os JWTs)
- Auditoria pós-incidente: query em AuditLog filtrado por `user_id`/`tenant_id`/janela
