import jwt
import pytest
from fastapi.websockets import WebSocketDisconnect

from app.security import AuthenticationError, authenticate


def test_valid_and_expired_jwt(config, token_factory):
    user = authenticate(token_factory(), config.jwt_secret.get_secret_value())
    assert (user.sub, user.department, user.level) == ("emp-001", "hr", 1)
    with pytest.raises(AuthenticationError):
        authenticate(token_factory(expired=True), config.jwt_secret.get_secret_value())
    with pytest.raises(AuthenticationError):
        authenticate(token_factory(), "different-secret-that-is-long-enough-for-hs256")
    with pytest.raises(AuthenticationError):
        authenticate("", config.jwt_secret.get_secret_value())


def test_wrong_role_claim_rejected(config):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {"sub": "abc", "department": "sales", "level": 0, "iat": now, "exp": now + timedelta(hours=1)},
        config.jwt_secret.get_secret_value(), algorithm="HS256"
    )
    with pytest.raises(AuthenticationError):
        authenticate(token, config.jwt_secret.get_secret_value())


@pytest.mark.parametrize("first", [
    {"type": "message", "text": "hello"},
    {"type": "auth", "token": "invalid"},
    {"type": "auth"},
])
def test_websocket_requires_valid_first_auth(client, first):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json(first)
        frame = ws.receive_json()
        assert frame["type"] == "auth_failed"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
