from sqlalchemy.ext.asyncio import AsyncSession
from app.models import AuditLog


async def write_audit(
    db: AsyncSession,
    *,
    action: str,
    user_id: str | None = None,
    tenant_id: str | None = None,
    resource_type: str = "",
    resource_id: str = "",
    ip_address: str = "",
    user_agent: str = "",
    metadata: dict | None = None,
    success: bool = True,
    notes: str = "",
) -> None:
    log = AuditLog(
        action=action,
        user_id=user_id,
        tenant_id=tenant_id,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        user_agent=user_agent,
        metadata_json=metadata or {},
        success="true" if success else "false",
        notes=notes,
    )
    db.add(log)
    await db.flush()
