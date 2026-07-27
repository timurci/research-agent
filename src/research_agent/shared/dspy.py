"""DSPy utilities.

Layer: Infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

from .metric import ToolCallObservation

if TYPE_CHECKING:
    from .config.models import LMConfig


def dspy_lm(lm_config: LMConfig) -> dspy.LM:
    """Build a DSPy LM from an ``LMConfig`` role mapping.

    Args:
        lm_config: Role fields (model, key, base URL, provider extras).

    Returns:
        Configured ``dspy.LM`` ready for ``dspy.settings.context``.
    """
    return dspy.LM(
        model=lm_config.model,
        api_key=lm_config.api_key,
        api_base=str(lm_config.base_url) if lm_config.base_url else None,
        extra_body=lm_config.provider_config,
    )


def extract_tool_calls(trajectory: dict[str, Any]) -> list[ToolCallObservation]:
    """Extract tool calls from a DSPy trajectory.

    Trajectory is a dictionary with the following pattern:

    ```
    {
        "thought_0": ...,
        "tool_name_0": ...,
        "tool_args_0": ...,
        "observation_0": ...,
        "thought_1": ...,
        "tool_name_1": ...,
        "tool_args_1": ...,
        "observation_1": ...,
        ...
    }
    ```

    The last tool called from a ReAct trajectory is named ``finish``.

    Args:
        trajectory: The DSPy trajectory to extract tool calls from.
    """
    tool_calls: list[ToolCallObservation] = []
    step = 0
    while True:
        name = trajectory.get(f"tool_name_{step}")
        args = trajectory.get(f"tool_args_{step}", {})
        obs = trajectory.get(f"observation_{step}")

        if name is None:
            break

        tool_calls.append(
            ToolCallObservation(tool_name=name, call_args=args, observation=obs)
        )

        step += 1
    return tool_calls
