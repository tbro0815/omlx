"""Installation method detection."""

import os
import shlex
import shutil
import sys
from pathlib import Path

_APP_BUNDLE_CLI_NAME = "omlx-cli"
_PATH_CLI = "omlx"
_USER_CLI_SHIM = Path(".omlx") / "bin" / "omlx"


def is_app_bundle() -> bool:
    """Return True if running inside the macOS .app bundle."""
    here = Path(__file__).resolve()
    return ".app/Contents/" in str(here)


def get_app_bundle_cli_path() -> Path:
    """Return the app-bundle CLI path for the currently running bundle."""
    here = Path(__file__).resolve()
    marker = ".app/Contents/"
    path = str(here)
    idx = path.find(marker)
    if idx == -1:
        return Path("/Applications/oMLX.app/Contents/MacOS") / _APP_BUNDLE_CLI_NAME
    app_root = Path(path[: idx + len(".app")])
    return app_root / "Contents" / "MacOS" / _APP_BUNDLE_CLI_NAME


def get_user_cli_shim_path() -> Path:
    """Return the user PATH shim installed by the macOS app."""
    return Path.home() / _USER_CLI_SHIM


def _is_executable(path: Path) -> bool:
    return path.exists() and os.access(path, os.X_OK)


def _same_resolved_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _is_app_managed_cli(path: Path) -> bool:
    """Return True when path points at the app-managed shim or wrapper."""
    if not _is_executable(path):
        return False
    user_shim = get_user_cli_shim_path()
    if _is_executable(user_shim) and _same_resolved_path(path, user_shim):
        return True
    app_cli = get_app_bundle_cli_path()
    return _is_executable(app_cli) and _same_resolved_path(path, app_cli)


def _path_resolves_to_app_managed_cli() -> bool:
    resolved = shutil.which(_PATH_CLI)
    return bool(resolved) and _is_app_managed_cli(Path(resolved))


def is_homebrew() -> bool:
    """Return True if running inside a Homebrew-installed virtualenv."""
    prefix = sys.prefix
    return "/Cellar/" in prefix or "/homebrew/" in prefix


def get_brew_formula_name(default: str = "omlx") -> str:
    """Return the Homebrew formula name this install came from.

    Read from the Cellar path (``/opt/homebrew/Cellar/<formula>/<version>/...``)
    rather than assumed, because the formula is not always called ``omlx``:
    the omni build installs as ``omlx-omni``, and a tap can carry several
    formulae side by side. Hardcoding ``omlx`` makes every generated command —
    ``brew services restart``, ``brew --prefix`` in install hints — fail with
    "Formula `omlx` is not installed" or, worse, silently address a *different*
    formula than the one that is running.

    Falls back to *default* for non-Homebrew installs (pip, .app bundle),
    where the value is only ever used to build a hint string.
    """
    parts = Path(sys.prefix).resolve().parts
    try:
        return parts[parts.index("Cellar") + 1]
    except (ValueError, IndexError):
        return default


def get_brew_prefix() -> str:
    """Return the Homebrew keg prefix (``.../Cellar/<formula>/<version>``).

    Empty string when this is not a Homebrew install.
    """
    resolved = Path(sys.prefix).resolve()
    parts = resolved.parts
    try:
        idx = parts.index("Cellar")
    except ValueError:
        return ""
    return str(Path(*parts[: idx + 3]))


def get_venv_pip_command() -> str:
    """Shell-safe path to the pip that installs into *this* environment.

    Used by "install X to use this feature" hints. On Homebrew that is the
    keg's own venv pip, addressed through ``brew --prefix <formula>`` so the
    command keeps working across version bumps.
    """
    if is_homebrew():
        return f'"$(brew --prefix {get_brew_formula_name()})/libexec/bin/pip"'
    return shlex.quote(str(Path(sys.prefix) / "bin" / "pip"))


def get_install_method() -> str:
    """Return the installation method: 'dmg', 'homebrew', or 'pip'."""
    if is_app_bundle():
        return "dmg"
    if is_homebrew():
        return "homebrew"
    return "pip"


def get_cli_prefix() -> str:
    """Return the correct CLI command prefix for the current installation."""
    if is_app_bundle():
        if _path_resolves_to_app_managed_cli():
            return _PATH_CLI
        return str(get_app_bundle_cli_path())
    return _PATH_CLI


def get_cli_command_prefix() -> str:
    """Return a shell-safe CLI command prefix for display/copy-paste."""
    return shlex.quote(get_cli_prefix())
