"""Generate signed expiring development JWTs compatible with provided test-client protocol."""
from datetime import datetime, timedelta, timezone

import jwt

from app.config import get_settings

USERS = (
    {"sub": "emp-001", "email": "emp@test.com", "department": "hr", "level": 1},
    {"sub": "mgr-002", "email": "mgr@test.com", "department": "hr", "level": 2},
    {"sub": "exec-003", "email": "exec@test.com", "department": "exec", "level": 3},
)


def main() -> None:
    secret = get_settings().jwt_secret.get_secret_value()
    if not secret:
        raise SystemExit("Set JWT_SECRET in .env first (must match server).")
    now = datetime.now(timezone.utc)
    for user in USERS:
        token = jwt.encode({**user, "iat": now, "exp": now + timedelta(hours=4)}, secret, algorithm="HS256")
        print(f"{user['sub']} ({user['department']}, level {user['level']}):\n{token}\n")


if __name__ == "__main__":
    main()
