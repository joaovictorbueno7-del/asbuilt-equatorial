from fastapi import APIRouter
from . import auth, users, agents, health, works, pipelines, dashboard, knowledge

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(works.router, prefix="/works", tags=["works"])
api_router.include_router(pipelines.router, prefix="/pipelines", tags=["pipelines"])
api_router.include_router(dashboard.router, tags=["dashboard"])
api_router.include_router(knowledge.router, prefix="/knowledge", tags=["knowledge"])
