"""Choice definition and Prompt.

A Choice question has a fixed, ordered set of Option IDs. A Prompt is Instructions plus one
Description per option. The Option IDs never change between Prompts; only the wording does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Choice:
    """A fixed-label Choice task.

    name: question name sent to Jev (question IDs are not shown to the model).
    option_ids: fixed Option IDs, in a fixed order. They are sent to the model as option names.
    state_field: key under which each row's text is placed in the request state.
    """

    name: str
    option_ids: tuple[str, ...]
    state_field: str = "text"

    def __post_init__(self) -> None:
        if len(set(self.option_ids)) != len(self.option_ids):
            raise ValueError(f"duplicate option ids: {self.option_ids}")
        if not self.option_ids:
            raise ValueError("a Choice needs at least one option")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "option_ids": list(self.option_ids), "state_field": self.state_field}


@dataclass(frozen=True)
class Prompt:
    """Instructions plus one Description per option (keyed by Option ID)."""

    instructions: str
    descriptions: dict[str, str] = field(default_factory=dict)

    def validate(self, choice: Choice) -> None:
        if set(self.descriptions) != set(choice.option_ids):
            missing = set(choice.option_ids) - set(self.descriptions)
            extra = set(self.descriptions) - set(choice.option_ids)
            raise ValueError(f"Prompt descriptions do not match options; missing={missing} extra={extra}")

    def criteria(self, choice: Choice) -> dict[str, str]:
        """Descriptions in the Choice's fixed option order."""
        self.validate(choice)
        return {oid: self.descriptions[oid] for oid in choice.option_ids}

    def to_dict(self) -> dict[str, Any]:
        return {"instructions": self.instructions, "descriptions": dict(self.descriptions)}


def build_request(choice: Choice, prompt: Prompt, text: str, model: str) -> dict[str, Any]:
    """The exact JSON body sent to POST /v1/systemone for one row."""
    return {
        "model": model,
        "state": {choice.state_field: text},
        "questions": {
            choice.name: {
                "type": "choice",
                "instructions": prompt.instructions,
                "criteria": prompt.criteria(choice),
            }
        },
    }


@dataclass(frozen=True)
class Arm:
    """A named way of answering a Choice: a fixed Prompt for now (later arms may add more)."""

    name: str  # e.g. "B3"
    slug: str  # e.g. "b3-minimal"
    choice: Choice
    prompt: Prompt
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "choice": self.choice.to_dict(),
            "prompt": self.prompt.to_dict(),
        }
