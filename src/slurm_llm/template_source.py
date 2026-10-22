"""Which file a job or setup template comes from, and keeping unedited copies up to date.

Your copies in ~/.config/slurm-llm/templates/ take precedence over the built-in templates. A copy
made by `sllm init --templates` that you never edited is not a choice to keep an old version: when
the built-in template changes (a new slurm-llm), such a copy is updated to it. A copy you edited is
kept and used, with one warning per new built-in version so the change is not missed.

A copy counts as unedited when its contents match the built-in template as it was when copied
(recorded by `init --templates` in .slurm-llm-templates.json) or any earlier built-in version
(builtin_versions.json, for copies made before that record existed).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import PACKAGE_DIR, ConfigError, tilde, user_template_dir

logger = logging.getLogger(__name__)

BUILTIN_DIR = PACKAGE_DIR / "templates"
RECORD = ".slurm-llm-templates.json"  # in your template directory
BUILTIN_VERSIONS = BUILTIN_DIR / "builtin_versions.json"


def fingerprint(path: Path) -> str:
    """sha256 of a file's contents (the first 12 hex digits identify a template version)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class TemplateSource:
    """The template file used for a name, and how it relates to the built-in one."""

    name: str
    path: Path
    yours: bool  # a file in your template directory, not the built-in one
    edited: bool = False  # yours, and not a copy of any built-in version
    has_builtin: bool = True  # a built-in template of this name exists
    updated: bool = False  # an unedited copy was just updated to the current built-in version
    outdated: bool = False  # an unedited copy of an older built-in version (not updated: update=False)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.path)[:12]

    def describe(self) -> str:
        """E.g. "built-in", "your copy ~/.config/slurm-llm/templates/vllm.slurm (edited)"."""
        if not self.yours:
            return "built-in"
        if not self.has_builtin:
            return f"your template {tilde(self.path)}"
        if self.edited:
            return f"your copy {tilde(self.path)} (edited)"
        if self.outdated:
            return f"your copy {tilde(self.path)} (unedited, of an older version; updated when next used)"
        return f"your copy {tilde(self.path)} (same as the built-in)"


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def builtin_versions(name: str) -> set[str]:
    """Fingerprints of every released version of a built-in template, including this one."""
    versions = set(_load_json(BUILTIN_VERSIONS).get(name) or [])
    if (BUILTIN_DIR / name).is_file():
        versions.add(fingerprint(BUILTIN_DIR / name))
    return versions


def record_copy(template_dir: Path, name: str) -> None:
    """Remember the built-in version a copy was made from (by `sllm init --templates`)."""
    path = template_dir / RECORD
    record = _load_json(path)
    record[name] = {"copied_from": fingerprint(BUILTIN_DIR / name)}
    path.write_text(json.dumps(record, indent=2) + "\n")


def resolve_template(name: str, template_dir: Path | None = None, update: bool = True) -> TemplateSource:
    """The template to use for `name`: your copy if there is one, else the built-in one.

    With `update`, an unedited copy of an older built-in version is replaced by the current one,
    and an edited copy logs a warning once per new built-in version.
    """
    template_dir = template_dir or user_template_dir()
    yours, builtin = template_dir / name, BUILTIN_DIR / name
    if not yours.is_file():
        if builtin.is_file():
            return TemplateSource(name, builtin, yours=False)
        available = sorted(p.name for p in BUILTIN_DIR.iterdir() if p.suffix in (".slurm", ".sh"))
        raise ConfigError(
            f"Template '{name}' not found in {template_dir} or the packaged templates "
            f"(available: {', '.join(available)})"
        )
    if not builtin.is_file():  # your own template, with no built-in counterpart
        return TemplateSource(name, yours, yours=True, has_builtin=False)

    current, copy = fingerprint(builtin), fingerprint(yours)
    if copy == current:
        return TemplateSource(name, yours, yours=True)
    record_path = template_dir / RECORD
    record = _load_json(record_path)
    entry = record.get(name) if isinstance(record.get(name), dict) else {}
    if copy in builtin_versions(name) | {entry.get("copied_from")}:
        if not update:
            return TemplateSource(name, yours, yours=True, outdated=True)
        yours.write_bytes(builtin.read_bytes())
        record[name] = {"copied_from": current}
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        logger.warning(
            f"Updated your copy of {name} ({tilde(yours)}) to the built-in version of this slurm-llm: "
            "you had not edited it"
        )
        return TemplateSource(name, yours, yours=True, updated=True)

    if update and entry.get("warned_for") != current:
        since = "has changed since you copied it" if entry.get("copied_from") else "differs from it"
        logger.warning(
            f"Using your edited {name} ({tilde(yours)}); the built-in {name} {since}. "
            f"Compare: diff {tilde(yours)} {builtin}"
        )
        record[name] = {**entry, "warned_for": current}
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    return TemplateSource(name, yours, yours=True, edited=True)
