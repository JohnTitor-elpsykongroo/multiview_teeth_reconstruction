from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from teeth3ds_dmm.reason_codes import ReasonCode


class Teeth3DSDMMError(RuntimeError):
    """Base error carrying a stable reason code and structured context."""

    def __init__(
        self,
        reason_code: ReasonCode,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
        self.context = dict(context or {})

    def __str__(self) -> str:
        context = ""
        if self.context:
            rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
            context = f" ({rendered})"
        return f"[{self.reason_code.value}] {self.message}{context}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code.value,
            "message": self.message,
            "context": self.context,
        }


class ConfigError(Teeth3DSDMMError):
    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(ReasonCode.E_CONFIG_INVALID, message, context=context)


class AtomicWriteError(Teeth3DSDMMError):
    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(ReasonCode.E_ATOMIC_WRITE_FAILED, message, context=context)


class CacheError(Teeth3DSDMMError):
    pass

