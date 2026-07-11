"""DSPy utilities.

Layer: Infrastructure.
"""

from typing import Any

from .metric import ToolCallObservation


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
