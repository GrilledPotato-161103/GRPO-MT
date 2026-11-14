"""Read a named secret without retaining it in configuration or diagnostics."""

import os
import sys


def environment_secret(name: str) -> str | None:
    """Read process variables, then Windows user variables for long-running shells."""
    value = os.environ.get(name)
    if value:
        return value
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
            return value if isinstance(value, str) and value else None
        except OSError:
            return None
    return None
