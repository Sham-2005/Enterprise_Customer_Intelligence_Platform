"""
Authentication & RBAC REST API Routes for ECIP.
Provides User Registration, JWT Login, and Role-Based Access Control hooks.
"""

import hashlib
import time
from typing import Dict, Any
from pydantic import BaseModel, Field, EmailStr
from fastapi import APIRouter, HTTPException, Depends

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & RBAC"])

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

class UserRegisterRequest(BaseModel):
    username: str = Field(..., example="admin_user")
    email: str = Field(..., example="admin@enterprise.com")
    password: str = Field(..., example="SuperSecret123!")
    role: str = Field("Business Analyst", example="Administrator")

class UserLoginRequest(BaseModel):
    username: str = Field(..., example="admin_user")
    password: str = Field(..., example="SuperSecret123!")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_seconds: int = 3600

# Mock DB store for users
MOCK_USERS_DB = {
    "admin_user": {
        "username": "admin_user",
        "email": "admin@enterprise.com",
        "hashed_password": hash_password("SuperSecret123!"),
        "role": "Administrator"
    }
}

@router.post("/register")
def register_user(request: UserRegisterRequest):
    if request.username in MOCK_USERS_DB:
        raise HTTPException(status_code=400, detail="Username already exists.")

    MOCK_USERS_DB[request.username] = {
        "username": request.username,
        "email": request.email,
        "hashed_password": hash_password(request.password),
        "role": request.role
    }

    return {"status": "success", "message": f"User '{request.username}' created with role '{request.role}'."}

@router.post("/login", response_model=TokenResponse)
def login_user(request: UserLoginRequest):
    user = MOCK_USERS_DB.get(request.username)
    if not user or user["hashed_password"] != hash_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # Generate token string
    token_str = f"ecip_jwt_token_{hashlib.md5(f'{request.username}:{time.time()}'.encode()).hexdigest()}"
    return TokenResponse(
        access_token=token_str,
        role=user["role"]
    )
