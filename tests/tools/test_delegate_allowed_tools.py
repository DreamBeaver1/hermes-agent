"""Tests for delegation.allowed_tools — the exact-tool ALLOWLIST for
delegated children (generic delegation security).

Contract under test:
  - Absent/None → existing behavior, byte-for-byte (backward compat).
  - A list (INCLUDING an empty one) → the child's final model-facing tool
    set is the INTERSECTION of:
      (a) tools the parent legitimately possesses,
      (b) the existing DELEGATE_BLOCKED_TOOLS / leaf restrictions,
      (c) exactly the allowlisted names.
  - The boundary lives at the tool-resolution layer
    (model_tools.get_tool_definitions → final subtraction) so registry,
    plugin, and MCP refreshes can never re-add a tool outside it.
  - Model/provider routing overrides (Qwen child routing, reasoning_effort,
    concurrency/depth) are unaffected.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import model_tools
from tools.delegate_tool import (
    DELEGATE_BLOCKED_TOOLS,
    _build_child_agent,
    _get_allowed_tools,
    _strip_blocked_tools,
)
from tools import mcp_tool


# The canonical hard read-only allowlist for a research/Qwen child.
READ_ONLY_ALLOWLIST = [
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "web_search",
    "web_extract",
]

MUTATION_TOOLS = ["write_file", "patch", "terminal", "process",
                  "execute_code", "skill_manage"]


def _tool(name):
    return {"type": "function",
            "function": {"name": name, "description": "", "parameters": {}}}


def _make_mock_parent(enabled_toolsets=None, valid_tool_names=None):
    """Mock parent agent with the fields _build_child_agent expects."""
    parent = MagicMock()
    parent.base_url = "https://api.nousresearch.com/v1"
    parent.api_key = "***"
    parent.provider = "nous"
    parent.api_mode = "chat_completions"
    parent.model = "z-ai/glm-5.3-flash"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = __import__("threading").Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = enabled_toolsets
    parent.disabled_toolsets = None
    parent.valid_tool_names = (
        set(valid_tool_names)
        if valid_tool_names is not None
        else set()
    )
    return parent


def _build_child_with_allowlist(parent, allowed_tools):
    """Build a child agent with delegation.allowed_tools patched in config.

    Returns the kwargs passed to AIAgent plus the REAL tool definitions the
    child's snapshot would resolve (exercising the true model_tools path,
    not a mock).
    """
    with (
        patch("run_agent.AIAgent") as MockAgent,
        patch("tools.delegate_tool._load_config",
              return_value={"allowed_tools": allowed_tools}),
    ):
        MockAgent.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="Read-only research",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=10,
            parent_agent=parent,
            task_count=1,
            role="leaf",
        )
    _, kwargs = MockAgent.call_args
    return kwargs


def _resolve_child_tools(kwargs):
    """Resolve the REAL model-facing tool set for the constructed child."""
    defs = model_tools.get_tool_definitions(
        enabled_toolsets=kwargs["enabled_toolsets"],
        disabled_toolsets=kwargs["disabled_toolsets"],
        allowed_tools=kwargs["allowed_tools"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    ) or []
    return {t["function"]["name"] for t in defs}


class TestGetAllowedToolsConfigParsing(unittest.TestCase):
    """Config parsing: absent / malformed / validated."""

    def test_absent_returns_none(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            self.assertIsNone(_get_allowed_tools())

    def test_non_list_fails_closed_not_silently_broadened(self):
        # A string like "read_file" must NOT be reinterpreted char-by-char
        # or as a single-name grant, and must NOT fall back to None (= no
        # boundary = broadening). It fails CLOSED to zero tools.
        with patch("tools.delegate_tool._load_config",
                   return_value={"allowed_tools": "read_file"}):
            self.assertEqual(_get_allowed_tools(), [])

    def test_unknown_names_dropped_with_warning(self):
        with (
            patch("tools.delegate_tool._load_config",
                  return_value={"allowed_tools":
                                ["read_file", "totally_fake_tool"]}),
            patch("tools.registry.registry.get_all_tool_names",
                  return_value=["read_file", "write_file", "terminal"]),
        ):
            result = _get_allowed_tools()
        self.assertEqual(result, ["read_file"])

    def test_blocked_names_dropped_even_if_allowlisted(self):
        blocked = sorted(DELEGATE_BLOCKED_TOOLS)
        with (
            patch("tools.delegate_tool._load_config",
                  return_value={"allowed_tools": ["read_file"] + blocked}),
            patch("tools.registry.registry.get_all_tool_names",
                  return_value=["read_file"] + blocked),
        ):
            result = _get_allowed_tools()
        self.assertEqual(result, ["read_file"])
        self.assertTrue(DELEGATE_BLOCKED_TOOLS.isdisjoint(result))

    def test_empty_list_is_preserved_not_coerced_to_none(self):
        """Empty allowlist must mean ZERO tools, never 'inherit all'."""
        with (
            patch("tools.delegate_tool._load_config",
                  return_value={"allowed_tools": []}),
            patch("tools.registry.registry.get_all_tool_names",
                  return_value=["read_file"]),
        ):
            self.assertEqual(_get_allowed_tools(), [])

    def test_malformed_explicit_value_fails_closed_to_empty(self):
        """PR #3 review follow-up: an explicit MALFORMED (non-list) value is
        a configuration error and must FAIL CLOSED — it must never resolve to
        ``None`` (= 'no exact-name boundary' = capability broadening)."""
        for malformed in ("read_file", 42, {"read_file": True}, 3.14, True):
            with self.subTest(malformed=malformed):
                with (
                    patch("tools.delegate_tool._load_config",
                          return_value={"allowed_tools": malformed}),
                    patch("tools.registry.registry.get_all_tool_names",
                          return_value=["read_file", "write_file", "terminal"]),
                ):
                    self.assertEqual(_get_allowed_tools(), [])

    def test_malformed_config_child_cannot_broaden_capability(self):
        """End-to-end: malformed explicit config → child kwargs carry an
        empty allowlist (zero tools), not None (inherited everything)."""
        parent = _make_mock_parent(
            enabled_toolsets=["file", "terminal", "web"],
            valid_tool_names=set(READ_ONLY_ALLOWLIST) | set(MUTATION_TOOLS),
        )
        kwargs = _build_child_with_allowlist(parent, "read_file")  # string
        self.assertEqual(kwargs["allowed_tools"], [])
        self.assertEqual(_resolve_child_tools(kwargs), set())

    def test_absent_config_still_means_none_backward_compat(self):
        """Guard the companion invariant: only a MALFORMED EXPLICIT value
        fails closed; absent/None config keeps returning None."""
        for absent in ({}, {"allowed_tools": None}):
            with self.subTest(config=absent):
                with patch("tools.delegate_tool._load_config",
                           return_value=absent):
                    self.assertIsNone(_get_allowed_tools())


class TestModelToolsAllowlistClamp(unittest.TestCase):
    """The final exact-name clamp inside get_tool_definitions."""

    def test_none_is_strict_noop(self):
        defs_no = model_tools.get_tool_definitions(
            enabled_toolsets=["file", "terminal"], quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        defs_none = model_tools.get_tool_definitions(
            enabled_toolsets=["file", "terminal"], quiet_mode=True,
            skip_tool_search_assembly=True, allowed_tools=None,
        )
        self.assertEqual(
            [t["function"]["name"] for t in defs_no],
            [t["function"]["name"] for t in defs_none],
        )

    def test_allowlist_restricts_toolset_expansion(self):
        names = {
            t["function"]["name"]
            for t in model_tools.get_tool_definitions(
                enabled_toolsets=["file", "terminal", "web"],
                quiet_mode=True, skip_tool_search_assembly=True,
                allowed_tools=["read_file", "web_search"],
            )
        }
        self.assertEqual(names, {"read_file", "web_search"})

    def test_empty_allowlist_yields_zero_tools(self):
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["file", "terminal", "web"],
            quiet_mode=True, skip_tool_search_assembly=True,
            allowed_tools=[],
        )
        self.assertEqual(defs, [])

    def test_allowlist_name_not_in_any_toolset_grants_nothing_extra(self):
        names = {
            t["function"]["name"]
            for t in model_tools.get_tool_definitions(
                enabled_toolsets=["file"],
                quiet_mode=True, skip_tool_search_assembly=True,
                allowed_tools=["read_file", "nuclear_launch"],
            )
        }
        self.assertEqual(names, {"read_file"})

    def test_clamp_survives_registry_refresh(self):
        """Rule 9/10: a registry generation bump (plugin/MCP tool landing)
        must not re-add non-allowlisted tools."""
        from tools.registry import registry

        kwargs = {
            "enabled_toolsets": ["file", "terminal"],
            "allowed_tools": ["read_file"],
            "quiet_mode": True,
            "skip_tool_search_assembly": True,
        }
        before = {
            t["function"]["name"]
            for t in model_tools.get_tool_definitions(**kwargs)
        }
        self.assertEqual(before, {"read_file"})

        # Simulate an MCP/plugin tool registering later in the child's
        # lifetime: the generation bumps and the cache key changes.
        entry = model_tools.registry.get_entry("read_file")
        self.assertIsNotNone(entry)
        fake_snapshot = [
            entry,
            SimpleNamespace(
                name="mcp_evil_write", toolset="mcp-x",
                check_fn=None,
                schema={"name": "mcp_evil_write",
                        "description": "", "parameters": {}},
            ),
        ]
        with patch.object(registry, "_snapshot_entries",
                          return_value=fake_snapshot):
            registry._generation += 1  # new registration → cache invalid
            after = {
                t["function"]["name"]
                for t in model_tools.get_tool_definitions(**kwargs)
            }
        self.assertEqual(after, {"read_file"})


class TestChildConstructionAllowlist(unittest.TestCase):
    """_build_child_agent threads the allowlist and intersects with parent."""

    def test_no_allowlist_child_kwargs_unchanged(self):
        parent = _make_mock_parent(enabled_toolsets=["file", "terminal"])
        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._load_config", return_value={}),
        ):
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="x", context=None, toolsets=None,
                model=None, max_iterations=10, parent_agent=parent,
                task_count=1, role="leaf",
            )
        _, kwargs = MockAgent.call_args
        self.assertIsNone(kwargs["allowed_tools"])

    def test_read_only_allowlist_reaches_child(self):
        parent = _make_mock_parent(
            enabled_toolsets=["file", "terminal", "web", "skills",
                              "session_search", "code_execution"],
            valid_tool_names=(
                set(READ_ONLY_ALLOWLIST) | set(MUTATION_TOOLS) | {"read_file"}
            ),
        )
        kwargs = _build_child_with_allowlist(parent, READ_ONLY_ALLOWLIST)
        self.assertEqual(sorted(kwargs["allowed_tools"]),
                         sorted(READ_ONLY_ALLOWLIST))

    def test_allowlisted_tool_parent_lacks_is_not_granted(self):
        parent = _make_mock_parent(
            enabled_toolsets=["file", "terminal", "web"],
            valid_tool_names=(
                set(READ_ONLY_ALLOWLIST) | set(MUTATION_TOOLS)
            ) - {"session_search"},  # parent lacks session_search
        )
        kwargs = _build_child_with_allowlist(parent, READ_ONLY_ALLOWLIST)
        self.assertNotIn("session_search", kwargs["allowed_tools"])

    def test_blocked_tools_stay_blocked_despite_allowlist(self):
        """Scenario 10: DELEGATE_BLOCKED_TOOLS remain authoritative."""
        parent = _make_mock_parent(
            enabled_toolsets=["file", "memory", "cronjob"],
            valid_tool_names=(
                set(READ_ONLY_ALLOWLIST) | {"memory", "cronjob", "write_file"}
            ),
        )
        # "memory"/"cronjob" are not in DELEGATE_BLOCKED_TOOLS... they ARE:
        # memory, cronjob, send_message, clarify, delegate_task. And the
        # config-layer validation also drops them (tested above). Here we
        # verify the end-to-end construction still passes no blocked tool.
        sneaky = READ_ONLY_ALLOWLIST + ["memory", "cronjob", "send_message"]
        kwargs = _build_child_with_allowlist(parent, sneaky)
        self.assertTrue(DELEGATE_BLOCKED_TOOLS.isdisjoint(
            kwargs["allowed_tools"]))

    def test_child_tool_resolution_end_to_end_read_only(self):
        """Scenarios 2-8 resolved through the REAL model_tools path."""
        parent = _make_mock_parent(
            enabled_toolsets=["file", "terminal", "web", "skills",
                              "session_search", "code_execution"],
            valid_tool_names=(
                set(READ_ONLY_ALLOWLIST) | set(MUTATION_TOOLS)
            ),
        )
        kwargs = _build_child_with_allowlist(parent, READ_ONLY_ALLOWLIST)
        names = _resolve_child_tools(kwargs)

        # Allowed read tools that exist in the registry are present.
        self.assertIn("read_file", names)
        self.assertIn("search_files", names)
        self.assertIn("web_search", names)
        self.assertIn("skill_view", names)
        # Mutation tools absent.
        for m in MUTATION_TOOLS:
            self.assertNotIn(m, names, f"{m} must not leak to the child")

    def test_empty_allowlist_child_gets_zero_tools(self):
        """Scenario 11: empty explicit list → zero model-facing tools."""
        parent = _make_mock_parent(
            enabled_toolsets=["file", "terminal"],
            valid_tool_names=set(READ_ONLY_ALLOWLIST) | set(MUTATION_TOOLS),
        )
        kwargs = _build_child_with_allowlist(parent, [])
        self.assertEqual(kwargs["allowed_tools"], [])
        self.assertEqual(_resolve_child_tools(kwargs), set())

    def test_empty_parent_tool_surface_keeps_child_empty(self):
        """PR #3 review follow-up: a parent whose ACTUAL resolved tool
        surface is the intentionally-empty set() must keep the child empty —
        broad enabled_toolsets plus a nonempty allowlist must NOT trigger the
        toolset-derived fallback (child ⊆ parent's real tools)."""
        parent = _make_mock_parent(
            enabled_toolsets=["file", "terminal", "web", "code_execution"],
            valid_tool_names=set(),  # exists and is intentionally EMPTY
        )
        self.assertEqual(parent.valid_tool_names, set())
        kwargs = _build_child_with_allowlist(parent, READ_ONLY_ALLOWLIST)
        self.assertEqual(kwargs["allowed_tools"], [])
        self.assertEqual(_resolve_child_tools(kwargs), set())

    def test_missing_parent_tool_names_attr_still_falls_back(self):
        """The companion case: when valid_tool_names is UNAVAILABLE (missing
        attribute / non-collection, e.g. MagicMock auto-attrs or None), the
        toolset-derived fallback intersection still applies."""
        parent = _make_mock_parent(
            enabled_toolsets=["file"],
            valid_tool_names=set(READ_ONLY_ALLOWLIST) | set(MUTATION_TOOLS),
        )
        parent.valid_tool_names = None  # attribute exists but is unusable
        kwargs = _build_child_with_allowlist(parent, READ_ONLY_ALLOWLIST)
        # Fallback derives from parent toolsets ("file"), so tools outside
        # that toolset are still dropped — intersection remains in force.
        self.assertNotIn("web_search", kwargs["allowed_tools"])
        self.assertNotIn("terminal", kwargs["allowed_tools"])

    def test_batch_children_receive_same_restriction(self):
        """Scenario 13: every child in a batch gets the identical allowlist."""
        parent = _make_mock_parent(
            enabled_toolsets=["file", "terminal", "web"],
            valid_tool_names=set(READ_ONLY_ALLOWLIST) | set(MUTATION_TOOLS),
        )
        kwargs = _build_child_with_allowlist(parent, READ_ONLY_ALLOWLIST)
        # _build_child_agent is invoked per task in batch mode (same code
        # path, task_index varies); the allowlist does not depend on the
        # index, so constructing again yields the identical restriction.
        kwargs2 = _build_child_with_allowlist(parent, READ_ONLY_ALLOWLIST)
        self.assertEqual(kwargs["allowed_tools"], kwargs2["allowed_tools"])

    def test_model_provider_override_unaffected(self):
        """Scenario 14: Qwen child routing must not regress."""
        parent = _make_mock_parent(
            enabled_toolsets=["file"],
            valid_tool_names=set(READ_ONLY_ALLOWLIST),
        )
        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._load_config",
                  return_value={"allowed_tools": READ_ONLY_ALLOWLIST}),
        ):
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="x", context=None, toolsets=None,
                model="qwen/qwen3.8-flash",
                max_iterations=10, parent_agent=parent, task_count=1,
                role="leaf", override_provider="nous",
                override_api_mode="chat_completions",
            )
        _, kwargs = MockAgent.call_args
        self.assertEqual(kwargs["model"], "qwen/qwen3.8-flash")
        self.assertEqual(kwargs["provider"], "nous")
        self.assertEqual(kwargs["api_mode"], "chat_completions")
        self.assertIsNotNone(kwargs["allowed_tools"])

    def test_reasoning_effort_override_unaffected(self):
        """Scenario 15: Qwen MEDIUM effort must not regress."""
        parent = _make_mock_parent(
            enabled_toolsets=["file"],
            valid_tool_names=set(READ_ONLY_ALLOWLIST),
        )
        parsed_effort = {"enabled": True, "effort": "medium"}
        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch(
                "tools.delegate_tool._load_config",
                return_value={"allowed_tools": READ_ONLY_ALLOWLIST,
                              "reasoning_effort": "medium"},
            ),
            patch("hermes_constants.parse_reasoning_effort",
                  return_value=parsed_effort),
        ):
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="x", context=None, toolsets=None,
                model=None, max_iterations=10, parent_agent=parent,
                task_count=1, role="leaf",
            )
        _, kwargs = MockAgent.call_args
        self.assertEqual(kwargs["reasoning_config"], parsed_effort)


class TestConcurrencyAndDepthUnchanged(unittest.TestCase):
    """Scenario 16: concurrency / depth knobs untouched by the allowlist."""

    def test_max_concurrent_children_and_depth_defaults(self):
        with patch("tools.delegate_tool._load_config",
                   return_value={"allowed_tools": READ_ONLY_ALLOWLIST}):
            from tools.delegate_tool import (
                _get_max_concurrent_children,
                _get_max_spawn_depth,
            )

            self.assertGreaterEqual(_get_max_concurrent_children(), 1)
            self.assertGreaterEqual(_get_max_spawn_depth(), 1)

    def test_orchestrator_role_still_grants_delegation_toolset(self):
        parent = _make_mock_parent(
            enabled_toolsets=["file"],
            valid_tool_names=set(READ_ONLY_ALLOWLIST) | {"delegate_task"},
        )
        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._load_config",
                  return_value={"allowed_tools": READ_ONLY_ALLOWLIST}),
            patch("tools.delegate_tool._get_orchestrator_enabled",
                  return_value=True),
            patch("tools.delegate_tool._get_max_spawn_depth",
                  return_value=2),
        ):
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="x", context=None, toolsets=None,
                model=None, max_iterations=10, parent_agent=parent,
                task_count=1, role="orchestrator",
            )
        _, kwargs = MockAgent.call_args
        self.assertIn("delegation", kwargs["enabled_toolsets"])


class TestMcpRefreshCannotReintroduce(unittest.TestCase):
    """Scenario 17: refresh_agent_mcp_tools must re-apply the boundary."""

    def _agent(self, names, allowed):
        a = SimpleNamespace()
        a.tools = [_tool(n) for n in names]
        a.valid_tool_names = set(names)
        a.enabled_toolsets = ["file"]
        a.disabled_toolsets = None
        a.allowed_tools = allowed
        a._context_engine_tool_names = set()
        return a

    def test_refresh_drops_late_landing_non_allowlisted_tool(self):
        agent = self._agent(["read_file"], ["read_file"])
        new_defs = [_tool("read_file"), _tool("terminal"),
                    _tool("mcp_new_server_tool")]
        import model_tools as mt
        with patch.object(mt, "get_tool_definitions", lambda **kw: new_defs):
            added = mcp_tool.refresh_agent_mcp_tools(agent)
        self.assertEqual(added, set())
        self.assertEqual({t["function"]["name"] for t in agent.tools},
                         {"read_file"})
        self.assertEqual(agent.valid_tool_names, {"read_file"})

    def test_refresh_with_none_allowlist_unchanged(self):
        agent = self._agent(["read_file"], None)
        new_defs = [_tool("read_file"), _tool("terminal")]
        import model_tools as mt
        with patch.object(mt, "get_tool_definitions", lambda **kw: new_defs):
            added = mcp_tool.refresh_agent_mcp_tools(agent)
        self.assertEqual(added, {"terminal"})
        self.assertEqual(agent.valid_tool_names, {"read_file", "terminal"})

    def test_refresh_reinject_clamped_for_injected_families(self):
        """A memory-provider tool that survives reinjection must still be
        clamped away when not allowlisted."""
        agent = self._agent(["read_file"], ["read_file"])
        agent._memory_manager = SimpleNamespace(
            get_all_tool_schemas=lambda: [
                {"name": "memory_search", "description": "", "parameters": {}}
            ]
        )
        new_defs = [_tool("read_file")]
        import model_tools as mt
        with patch.object(mt, "get_tool_definitions", lambda **kw: new_defs), \
             patch("agent.memory_manager.memory_provider_tools_enabled",
                   return_value=True):
            mcp_tool.refresh_agent_mcp_tools(agent)
        names = {t["function"]["name"] for t in agent.tools}
        self.assertEqual(names, {"read_file"})


class TestBackwardCompatibility(unittest.TestCase):
    """Scenario 1: no allowlist configured → existing behavior unchanged."""

    def test_strip_blocked_tools_still_authoritative(self):
        result = _strip_blocked_tools(["terminal", "delegation", "clarify",
                                       "memory", "cronjob"])
        self.assertEqual(sorted(result), ["terminal"])

    def test_child_construction_without_config_matches_legacy_kwargs(self):
        parent = _make_mock_parent(enabled_toolsets=["file", "terminal"])
        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._load_config", return_value={}),
        ):
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="x", context=None, toolsets=None,
                model=None, max_iterations=10, parent_agent=parent,
                task_count=1, role="leaf",
            )
        _, kwargs = MockAgent.call_args
        # None → get_tool_definitions filtering identical to today.
        defs_legacy = model_tools.get_tool_definitions(
            enabled_toolsets=kwargs["enabled_toolsets"],
            disabled_toolsets=kwargs["disabled_toolsets"],
            quiet_mode=True, skip_tool_search_assembly=True,
        )
        defs_now = model_tools.get_tool_definitions(
            enabled_toolsets=kwargs["enabled_toolsets"],
            disabled_toolsets=kwargs["disabled_toolsets"],
            allowed_tools=None,
            quiet_mode=True, skip_tool_search_assembly=True,
        )
        self.assertEqual(
            [t["function"]["name"] for t in defs_legacy],
            [t["function"]["name"] for t in defs_now],
        )


class TestForgePrimaryUnaffected(unittest.TestCase):
    """Proof Forge primary (the parent) has zero capability change: the
    allowlist is read only when constructing CHILDREN; the parent's own
    toolsets/enabled tools never consult it."""

    def test_parent_toolset_attributes_not_mutated_by_child_build(self):
        parent = _make_mock_parent(
            enabled_toolsets=["file", "terminal", "code_execution"],
            valid_tool_names=set(READ_ONLY_ALLOWLIST) | set(MUTATION_TOOLS),
        )
        before = list(parent.enabled_toolsets)
        before_names = set(parent.valid_tool_names)
        _build_child_with_allowlist(parent, READ_ONLY_ALLOWLIST)
        self.assertEqual(list(parent.enabled_toolsets), before)
        self.assertEqual(set(parent.valid_tool_names), before_names)

    def test_primary_resolution_ignores_allowlist_when_none_passed(self):
        # The parent's own snapshot path passes allowed_tools=None — the
        # exact call shape every non-delegation caller uses.
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["file", "terminal", "code_execution"],
            quiet_mode=True, skip_tool_search_assembly=True,
        )
        names = {t["function"]["name"] for t in defs}
        self.assertIn("write_file", names)
        self.assertIn("patch", names)
        self.assertIn("terminal", names)
        self.assertIn("execute_code", names)


if __name__ == "__main__":
    unittest.main()
