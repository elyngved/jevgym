import json

import httpx
import pytest

from jevgym.choice import Arm, Choice, Prompt
from jevgym.data import Row

CHOICE = Choice(name="answer_type", option_ids=("ABBR", "DESC", "NUM"), state_field="question")
ARM = Arm(
    name="BX",
    slug="bx-test",
    choice=CHOICE,
    prompt=Prompt("What kind of answer does `question` ask for?", {"ABBR": "abbreviation", "DESC": "description", "NUM": "numeric"}),
)
ROWS = [
    Row("r1", "How many feet in a mile ?", "NUM"),
    Row("r2", "What does NASA stand for ?", "ABBR"),
    Row("r3", "Why is the sky blue ?", "DESC"),
    Row("r4", "How far is the moon ?", "NUM"),
]


def fake_transport(model="jev-1.14.0", pick=None, fail_status=None):
    """Answers every Choice with a fixed distribution; `pick` maps question text -> top option."""
    pick = pick or {}

    def handler(request: httpx.Request) -> httpx.Response:
        if fail_status:
            return httpx.Response(fail_status, json={"detail": "nope"})
        body = json.loads(request.content)
        text = body["state"]["question"]
        (qname, q), = body["questions"].items()
        opts = list(q["criteria"])
        top = pick.get(text, opts[0])
        probs = {o: (0.7 if o == top else 0.3 / (len(opts) - 1)) for o in opts}
        return httpx.Response(
            200,
            json={
                "model": model,
                "answers": {qname: {"type": "choice", "choice": top, "confidence": 0.6, "probabilities": probs}},
                "usage": {"input_tokens": 50, "output_tokens": 3},
            },
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def arm():
    return ARM


@pytest.fixture
def rows():
    return list(ROWS)
