import asyncio

import pytest

from conftest import fake_transport
from jevgym.choice import build_request
from jevgym.jev import PIN, JevAPIError, JevClient, JevError, PinMismatchError, cost_for


def ask(transport, request, **kw):
    async def go():
        async with JevClient(api_key="test", transport=transport, **kw) as c:
            return await c.system_one("r1", request)

    return asyncio.run(go())


def test_pinned_call_records_cost(arm):
    rec = ask(fake_transport(), build_request(arm.choice, arm.prompt, "Why ?", PIN))
    assert rec.returned_model == PIN and rec.pin_ok
    assert rec.input_tokens == 50 and rec.cost_usd == pytest.approx(cost_for(50)) == pytest.approx(50 * 42e-9)
    assert rec.request["model"] == PIN


def test_model_mismatch_raises_with_call_attached(arm):
    with pytest.raises(PinMismatchError) as ei:
        ask(fake_transport(model="jev-1.15.0"), build_request(arm.choice, arm.prompt, "Why ?", PIN))
    assert ei.value.call.returned_model == "jev-1.15.0" and not ei.value.call.pin_ok


def test_alias_in_response_is_a_mismatch(arm):
    with pytest.raises(PinMismatchError):
        ask(fake_transport(model="jev-latest"), build_request(arm.choice, arm.prompt, "Why ?", PIN))


def test_request_must_use_pin(arm):
    with pytest.raises(JevError):
        ask(fake_transport(), build_request(arm.choice, arm.prompt, "Why ?", "jev-latest"))


def test_http_error_is_an_error_not_an_answer(arm):
    with pytest.raises(JevAPIError) as ei:
        ask(fake_transport(fail_status=422), build_request(arm.choice, arm.prompt, "Why ?", PIN))
    assert ei.value.call.attempts == 1  # 422 is not retried


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError):
        JevClient()
