"""Bash tool for shell command execution."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import re

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm

from ..const import (
    DEFAULT_WORKING_DIRECTORY,
    SHELL_DENY_PATTERNS,
    SHELL_OUTPUT_LIMIT,
    SHELL_TIMEOUT,
)
from .base import Function

_LOGGER = logging.getLogger(__name__)


class BashFunction(Function):
    """Execute shell commands with security controls."""

    def get_working_dir(self, hass: HomeAssistant) -> Path:
        """Get the default working directory for bash operations."""
        return Path(hass.config.config_dir) / DEFAULT_WORKING_DIRECTORY

    def __init__(self) -> None:
        """Initialize bash tool."""
        schema = vol.Schema(
            {
                vol.Required("command"): cv.template,
                vol.Optional("cwd"): cv.template,
                vol.Optional("restrict_to_workspace", default=True): bool,
                vol.Optional("allow_patterns"): vol.All(cv.ensure_list, [str]),
            }
        )
        super().__init__(schema)

    def _guard_command(
        self,
        command: str,
        cwd: str | Path,
        restrict_to_workspace: bool,
        allow_patterns: list[str] | None = None,
    ) -> None:
        """Validate command against security policies."""
        # Deny patterns check
        for pattern in SHELL_DENY_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                raise ValueError(
                    f"Command blocked by security policy: matches pattern '{pattern}'"
                )

        # Allow patterns check
        if allow_patterns:
            lower = command.lower()
            if not any(re.search(p, lower) for p in allow_patterns):
                raise ValueError("Command blocked: not in allowlist")

        # Path restriction check when restrict_to_workspace is enabled
        if restrict_to_workspace:
            # Block path traversal patterns
            if "../" in command or "..\\" in command:
                raise ValueError("Command blocked: path traversal detected")

            # Extract and validate paths in command
            win_paths = re.findall(r"[A-Za-z]:\\[^\\\"\' ]+", command)
            posix_paths = re.findall(r"(?<!\w)/[^\s\"\']+", command)

            for raw in win_paths + posix_paths:
                try:
                    p = Path(raw).resolve()
                except Exception:
                    continue

                if cwd not in p.parents and p != cwd:
                    raise ValueError(
                        f"Command blocked by safety guard (path '{raw}' outside working dir).\nSet 'restrict_to_workspace: false' to allow command outside working directory."
                    )

    async def execute(
        self,
        hass: HomeAssistant,
        function_config,
        arguments,
        llm_context: llm.LLMContext | None,
        exposed_entities,
    ):
        """Execute shell command with security controls."""
        # Render command template
        command_template = function_config.get("command")
        command = command_template.async_render(arguments, parse_result=False)

        # Render cwd template if provided
        cwd_template = function_config.get("cwd")
        if cwd_template:
            cwd = Path(cwd_template.async_render(arguments, parse_result=False))
        else:
            cwd = self.get_working_dir(hass)

        timeout = arguments.get("timeout", SHELL_TIMEOUT)
        restrict_to_workspace = function_config.get("restrict_to_workspace", True)
        allow_patterns = function_config.get("allow_patterns", [])

        # Security validation
        try:
            self._guard_command(
                command,
                cwd=cwd,
                restrict_to_workspace=restrict_to_workspace,
                allow_patterns=allow_patterns,
            )
        except ValueError as err:
            return {"error": str(err)}

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except TimeoutError:
                process.kill()
                return {"error": f"Command timed out after {timeout} seconds"}

            # Decode output with truncation
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

            # Truncate output if too large
            if len(stdout_text) > SHELL_OUTPUT_LIMIT:
                stdout_text = (
                    stdout_text[:SHELL_OUTPUT_LIMIT]
                    + "\n... (truncated, output too large)"
                )
            if len(stderr_text) > SHELL_OUTPUT_LIMIT:
                stderr_text = (
                    stderr_text[:SHELL_OUTPUT_LIMIT]
                    + "\n... (truncated, output too large)"
                )

            result = {
                "exit_code": process.returncode,
                "stdout": stdout_text,
            }

            if stderr_text:
                result["stderr"] = stderr_text

        except Exception as e:
            _LOGGER.error(e)
            return {"error": str(e)}

        return result
