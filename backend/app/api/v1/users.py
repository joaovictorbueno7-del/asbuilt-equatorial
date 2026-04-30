from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.models import User
from app.auth.dependencies import get_current_user

router = APIRouter()


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    tenant_id: str
    mfa_enabled: bool
    is_active: bool


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return UserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        tenant_id=user.tenant_id,
        mfa_enabled=user.mfa_enabled,
        is_active=user.is_active,
    )
