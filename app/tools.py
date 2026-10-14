"""The single exposed tool, internally gathering three mocked services concurrently."""
import asyncio


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_employee_context",
        "description": (
            "Retrieve the authenticated employee's basic profile (including grade), manager, "
            "and team. Use for personal questions such as 'who is my manager' or "
            "'given my grade'. Does NOT return personal leave balance."
        ),
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "description": "The authenticated user's ID"}},
            "required": ["user_id"],
            "additionalProperties": False,
        },
    },
}


async def _profile(user_id: str) -> dict:
    await asyncio.sleep(1)
    return {"name": "Jane Doe", "grade": "Senior"}


async def _manager(user_id: str) -> dict:
    await asyncio.sleep(1)
    return {"manager": "John Smith"}


async def _team(user_id: str) -> dict:
    await asyncio.sleep(1)
    return {"team_size": 8, "team_name": "Platform"}


async def get_employee_context(user_id: str) -> dict:
    """Fan out with gather: about one second, not three."""
    profile, manager, team = await asyncio.gather(
        _profile(user_id), _manager(user_id), _team(user_id)
    )
    return {"user_id": user_id, **profile, **manager, **team}
