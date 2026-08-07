"""Query visibility and cache lifecycle rules."""
from __future__ import annotations

from kvbench.config import ProtocolConfig
from kvbench.errors import ProtocolError


class Protocol:
    def __init__(self, cfg: ProtocolConfig):
        self.cfg = cfg

    @property
    def query_visible(self) -> bool:
        return self.cfg.visibility == "query_visible"

    @property
    def live_bounded(self) -> bool:
        return self.cfg.cache_mode == "live_bounded"

    def validate_method(self, requires_visible_query: bool, method_name: str) -> None:
        if requires_visible_query and not self.query_visible:
            raise ProtocolError(
                "method=%s requires a query-visible current request, but protocol is query_agnostic"
                % method_name
            )

    def should_compress_prefill(self) -> bool:
        return True

    def should_compress_before_decode(self, step: int, cache_len: int, budget: int) -> bool:
        if not self.live_bounded:
            return False
        return cache_len + 1 > budget

    def should_recompute(self, step: int) -> bool:
        policy = self.cfg.update_policy
        if policy == "prefill_once":
            return False
        if policy == "every_step":
            return True
        if policy == "periodic":
            return step > 0 and step % int(self.cfg.update_interval) == 0
        raise ProtocolError("unknown update policy: %s" % policy)
