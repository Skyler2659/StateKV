"""Typed errors used to fail research runs explicitly."""


class KVBenchError(RuntimeError):
    """Base error for the Torch benchmark."""


class ConfigurationError(KVBenchError):
    """Raised when a resolved experiment configuration is invalid."""


class BudgetError(KVBenchError):
    """Raised when a method cannot satisfy the declared cache budget."""


class ProtocolError(KVBenchError):
    """Raised when a method would violate query-visibility or lifecycle rules."""


class SignalUnavailableError(KVBenchError):
    """Raised instead of silently substituting another importance signal."""


class UnsupportedMethodError(KVBenchError):
    """Raised when a method is unavailable for the selected backend/protocol."""

