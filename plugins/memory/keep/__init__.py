"""Keep memory plugin — typed delegate to keep.hermes.KeepMemoryProvider.

Keep provides reflective memory with semantic search, tagging, versioned
session capture, and agent prompts.  All logic lives in the ``keep-skill``
package (``keep/hermes/provider.py``).  This module is a thin typed
wrapper that inherits from the Hermes ``MemoryProvider`` ABC so that:

  - Hermes plugin discovery finds a proper ``MemoryProvider`` subclass
  - The keep-side code remains free of Hermes dependencies

Integration model:
  - ``keep-skill`` manages an in-process Keeper for reads and writes
  - A background daemon (auto-started) handles embeddings and summaries
  - The Keeper is created once in ``initialize()`` and closed in
    ``shutdown()``; it is held for the entire Hermes session

Setup:
  - ``pip install keep-skill`` (or ``uv pip install keep-skill``)
  - ``hermes memory setup`` → select ``keep``
  - Store defaults to ``$HERMES_HOME/keep``; override with
    ``KEEP_STORE_PATH``
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_KEEP_FLOW_CONTROL_KEYS = {"state", "params", "token_budget", "cursor", "state_doc_yaml", "budget"}


def _coerce_params(value: Any) -> Dict[str, Any]:
    """Accept dict or JSON-string params from chat/tool surfaces."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _raw_keep_results_enabled() -> bool:
    value = os.getenv("HERMES_KEEP_RAW_RESULTS") or os.getenv("HERMES_MEMORY_TOOLS_RAW")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility normalization at the keep plugin boundary.

    Hermes should keep passing the canonical schema, but this layer makes the
    keep plugin resilient to chat surfaces that flatten flow params or turn the
    params object into a JSON string.
    """
    if not isinstance(args, dict):
        return {}

    normalized = dict(args)
    if tool_name != "keep_flow":
        return normalized

    params = _coerce_params(normalized.get("params"))
    for key, value in args.items():
        if key not in _KEEP_FLOW_CONTROL_KEYS:
            params.setdefault(key, value)
    if params:
        normalized["params"] = params

    if _raw_keep_results_enabled() and "token_budget" not in normalized:
        normalized["token_budget"] = 0

    return normalized


def _load_impl():
    """Lazy-load the keep-side provider. Returns instance or None."""
    try:
        from keep.hermes import KeepMemoryProvider as _Impl
        return _Impl()
    except ImportError:
        return None


class KeepMemoryProvider(MemoryProvider):
    """Typed delegate that forwards every call to the keep-side provider.

    The keep-side import is deferred so this class can be instantiated
    even when keep-skill is not installed.  This allows Hermes plugin
    discovery to list keep in the setup wizard, where pip install is
    triggered before the provider is actually used.
    """

    def __init__(self):
        self._impl = _load_impl()

    def _ensure_impl(self):
        """Retry loading if the initial import failed (e.g. after pip install)."""
        if self._impl is None:
            self._impl = _load_impl()
        return self._impl is not None

    # -- Identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "keep"

    # -- Availability & config -----------------------------------------------

    def is_available(self) -> bool:
        if not self._ensure_impl():
            return False
        return self._impl.is_available()

    def get_config_schema(self) -> List[Dict[str, Any]]:
        if not self._ensure_impl():
            return [
                {
                    "key": "_keep_not_installed",
                    "description": "keep-skill is not installed",
                    "choices": [],
                    "empty_message": "keep-skill must be installed before configuring.",
                    "empty_hints": [
                        "pip install keep-skill",
                        "uv pip install keep-skill",
                    ],
                },
            ]
        return self._impl.get_config_schema()

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        if not self._ensure_impl():
            raise ValueError(
                "keep-skill is not installed. "
                "Run `pip install keep-skill` or `uv pip install keep-skill` "
                "and rerun `hermes memory setup`."
            )
        self._impl.save_config(values, hermes_home)

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        if not self._ensure_impl():
            logger.warning("keep: impl not available, skipping initialize")
            return
        self._impl.initialize(session_id, **kwargs)
        logger.info("keep: initialized (keeper=%s)", self._impl._keeper is not None)

    def shutdown(self) -> None:
        if self._impl is not None:
            self._impl.shutdown()

    # -- System prompt & recall ----------------------------------------------

    def system_prompt_block(self) -> str:
        if self._impl is None:
            return ""
        return self._impl.system_prompt_block()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._impl is None:
            return ""
        return self._impl.prefetch(query, session_id=session_id)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._impl is not None:
            self._impl.queue_prefetch(query, session_id=session_id)

    # -- Turn sync -----------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        if self._impl is not None:
            self._impl.sync_turn(user_content, assistant_content, session_id=session_id)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        if self._impl is not None:
            self._impl.on_turn_start(turn_number, message, **kwargs)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._impl is not None:
            self._impl.on_session_end(messages)

    # -- Observation hooks ---------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if self._impl is not None:
            self._impl.on_memory_write(action, target, content)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if self._impl is None:
            return ""
        return self._impl.on_pre_compress(messages)

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        if self._impl is not None:
            self._impl.on_delegation(
                task, result, child_session_id=child_session_id, **kwargs
            )

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Return schemas even before initialize — hermes indexes tool names
        # at add_provider time, before initialize is called.
        if self._impl is None:
            try:
                from keep.hermes.const import FLOW_SCHEMA, HELP_SCHEMA, PROMPT_SCHEMA
                schemas = [FLOW_SCHEMA, HELP_SCHEMA, PROMPT_SCHEMA]
                logger.info("keep: get_tool_schemas (pre-init): %d tools", len(schemas))
                return schemas
            except ImportError:
                logger.info("keep: get_tool_schemas: keep-skill not importable")
                return []
        schemas = self._impl.get_tool_schemas()
        logger.info("keep: get_tool_schemas: %d tools", len(schemas))
        return schemas

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        normalized_args = _normalize_tool_args(tool_name, args)
        logger.info("keep: handle_tool_call(%s)", tool_name)
        logger.debug("keep: raw args for %s: %s", tool_name, args)
        if normalized_args != args:
            logger.debug("keep: normalized args for %s: %s", tool_name, normalized_args)
        if self._impl is None:
            logger.warning("keep: handle_tool_call but impl is None")
            return '{"error": "keep-skill is not installed"}'
        result = self._impl.handle_tool_call(tool_name, normalized_args, **kwargs)
        logger.debug("keep: raw result for %s: %s", tool_name, result)
        logger.info("keep: handle_tool_call(%s) -> %s", tool_name, result[:200])
        return result


# ---------------------------------------------------------------------------
# Plugin entry point — called by Hermes memory plugin discovery
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register Keep as a memory provider plugin."""
    ctx.register_memory_provider(KeepMemoryProvider())
