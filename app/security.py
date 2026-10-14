"""JWT authentication. The client never gets to pick its own role or user ID."""
import jwt
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal


class User(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    sub: str = Field(min_length=1)
    department: Literal["hr", "finance", "exec"]
    level: int = Field(ge=1, le=3)
    email: str | None = None
    iat: int
    exp: int


class AuthenticationError(Exception):
    """Token was missing, untrusted, expired, or had an invalid claim."""


def authenticate(token: str, secret: str) -> User:
    if not token or not secret:
        raise AuthenticationError("Authentication token is missing or invalid.")
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"require": ["sub", "department", "level", "iat", "exp"]},
            leeway=0,
        )
        return User.model_validate(claims)
    except (jwt.PyJWTError, ValidationError, TypeError, ValueError) as exc:
        raise AuthenticationError("Authentication token is invalid or expired.") from exc
