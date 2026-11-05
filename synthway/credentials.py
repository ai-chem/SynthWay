"""One explicit credential-file policy for the GUI, CLI and demo.

Only the first existing project .env is read. No ancestor-directory search is
performed, and credential values are never logged here.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def credential_file(root=ROOT):
    root = Path(root)
    return next((path for path in (root / "synthway/.env", root / ".env") if path.is_file()), None)


def credential_environment(root=ROOT, environ=None):
    from dotenv import dotenv_values

    env = dict(os.environ if environ is None else environ)
    selected = credential_file(root)
    if selected is not None:
        env.update(
            {key: value for key, value in dotenv_values(selected).items() if value is not None}
        )
    return env, selected


def load_credentials(root=ROOT):
    env, selected = credential_environment(root)
    os.environ.update(env)
    return selected


def credential_guidance(root=ROOT):
    root = Path(root)
    selected = credential_file(root)
    source = (
        f"Credential file:\n{selected}"
        if selected
        else (
            f"No project credential file found. Create:\n{root / '.env'}\n"
            "Exported environment variables can also supply credentials."
        )
    )
    return (
        source + "\n\nsynthway/.env takes precedence over the root .env.\n"
        "The selected file overrides exported values; keep only one active file.\n"
        "Restart the application after changing credentials."
    )
