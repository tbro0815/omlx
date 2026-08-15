# SPDX-License-Identifier: Apache-2.0
"""Homebrew formula-name detection.

The formula is not always called ``omlx`` — the omni build installs as
``omlx-omni``. Everything that generates a brew command (``omlx restart``, the
"install apple-fm-sdk with ..." hints) used to hardcode ``omlx``, which fails
with "Formula `omlx` is not installed" or, if a tap happens to carry a formula
by that name, quietly addresses the wrong one.
"""

from __future__ import annotations

import sys

import pytest

from omlx.utils import install


@pytest.fixture
def prefix(monkeypatch):
    def _set(value: str):
        monkeypatch.setattr(sys, "prefix", value)

    return _set


@pytest.mark.parametrize(
    "sys_prefix,expected",
    [
        ("/opt/homebrew/Cellar/omlx-omni/0.5.8.dev3-omni/libexec", "omlx-omni"),
        ("/opt/homebrew/Cellar/omlx/0.4.4/libexec", "omlx"),
        ("/usr/local/Cellar/omlx-omni/1.0/libexec", "omlx-omni"),
    ],
)
def test_formula_name_comes_from_the_cellar_path(prefix, sys_prefix, expected):
    prefix(sys_prefix)
    assert install.get_brew_formula_name() == expected


def test_non_homebrew_prefix_falls_back_to_the_default(prefix):
    prefix("/usr/local/venvs/omlx-dev")
    assert install.get_brew_formula_name() == "omlx"
    assert install.get_brew_formula_name(default="other") == "other"
    assert install.get_brew_prefix() == ""


def test_brew_prefix_is_the_keg_root(prefix):
    prefix("/opt/homebrew/Cellar/omlx-omni/0.5.8.dev3-omni/libexec")
    assert install.get_brew_prefix() == (
        "/opt/homebrew/Cellar/omlx-omni/0.5.8.dev3-omni"
    )


def test_pip_hint_addresses_the_running_formula(prefix):
    prefix("/opt/homebrew/Cellar/omlx-omni/0.5.8.dev3-omni/libexec")
    hint = install.get_venv_pip_command()
    assert hint == '"$(brew --prefix omlx-omni)/libexec/bin/pip"'
    # The bug this guards: a hint that sends the user to a formula they do
    # not have installed.
    assert "brew --prefix omlx)" not in hint


def test_pip_hint_outside_homebrew_points_at_the_active_venv(prefix):
    prefix("/usr/local/venvs/omlx-dev")
    assert install.get_venv_pip_command().endswith("/usr/local/venvs/omlx-dev/bin/pip")


def test_brew_services_uses_the_detected_formula(prefix, monkeypatch):
    """``omlx restart`` must not shell out to ``brew services restart omlx``."""
    import omlx.cli as cli

    prefix("/opt/homebrew/Cellar/omlx-omni/0.5.8.dev3-omni/libexec")
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    # _run_brew_services imports shutil/subprocess inside the function.
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/brew")
    monkeypatch.setattr("subprocess.run", lambda argv, **kw: calls.append(argv) or _Result())

    assert cli._run_brew_services("restart") == 0
    assert calls == [["/opt/homebrew/bin/brew", "services", "restart", "omlx-omni"]]
