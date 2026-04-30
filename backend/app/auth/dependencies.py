from datetime import datetime, timezone
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.security import decode_token
from app.models import User, UserRole

bearer = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    try:
        payload = decode_token(credentials.credentials)
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    if payload.get("type") != "access":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong token type")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Malformed token")

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User inactive or not found")
    if user.locked_until and user.locked_until > datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_423_LOCKED, "Account locked")
    return user


def require_roles(*roles: UserRole):
    allowed = {r.value for r in roles}

    async def checker(user: User = Depends(get_current_user)) -> User:
        if user.role.value not in allowed and user.role != UserRole.ADMIN:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Requires one of roles: {sorted(allowed)}",
            )
        return user

    return checker


async def get_client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""
