"""Language model configuration.

Layer: Application.
"""

from pydantic import BaseModel, ConfigDict, HttpUrl


class LMConfig(BaseModel):
    """Language model configuration.

    Assuming LiteLLM conventions in model names.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    api_key: str | None = None
    base_url: HttpUrl | None = None
