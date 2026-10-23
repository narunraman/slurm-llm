"""Which template a job uses, and keeping unedited copies up to date (issue #1)."""

import json
import shutil

import pytest

from slurm_llm import template_source
from slurm_llm.config import PACKAGE_DIR, ConfigError
from slurm_llm.template_source import builtin_versions, fingerprint, record_copy, resolve_template


@pytest.fixture
def builtin(tmp_path, monkeypatch):
    """A copy of the built-in templates that a test can "upgrade"."""
    d = tmp_path / "builtin"
    shutil.copytree(PACKAGE_DIR / "templates", d)
    monkeypatch.setattr(template_source, "BUILTIN_DIR", d)
    monkeypatch.setattr(template_source, "BUILTIN_VERSIONS", d / "builtin_versions.json")
    return d


@pytest.fixture
def mine(tmp_path):
    d = tmp_path / "mine"
    d.mkdir()
    return d


def _copy(builtin, mine, name="vllm.slurm"):
    """What `sllm init --templates` does."""
    shutil.copy(builtin / name, mine / name)
    record_copy(mine, name)


def _upgrade(builtin, name="vllm.slurm"):
    """A new slurm-llm with a changed built-in template."""
    (builtin / name).write_text((builtin / name).read_text() + "# fixed in a later version\n")


def test_without_a_copy_the_built_in_is_used(builtin, mine):
    source = resolve_template("vllm.slurm", mine)
    assert (source.path, source.yours, source.describe()) == (builtin / "vllm.slurm", False, "built-in")
    with pytest.raises(ConfigError, match="not found"):
        resolve_template("missing.slurm", mine)


def test_an_unedited_copy_follows_the_built_in(builtin, mine, caplog):
    _copy(builtin, mine)
    assert "same as the built-in" in resolve_template("vllm.slurm", mine).describe()
    _upgrade(builtin)
    source = resolve_template("vllm.slurm", mine)
    assert source.updated and source.path == mine / "vllm.slurm"
    assert (mine / "vllm.slurm").read_text() == (builtin / "vllm.slurm").read_text()
    assert "Updated your copy of vllm.slurm" in caplog.text and "you had not edited it" in caplog.text


def test_an_edited_copy_is_kept_with_one_warning_per_built_in_version(builtin, mine, caplog):
    _copy(builtin, mine)
    (mine / "vllm.slurm").write_text((mine / "vllm.slurm").read_text() + "#SBATCH --partition=mine\n")
    edited = (mine / "vllm.slurm").read_text()
    _upgrade(builtin)
    for _ in range(3):
        source = resolve_template("vllm.slurm", mine)
        assert source.edited and source.path == mine / "vllm.slurm"
    assert (mine / "vllm.slurm").read_text() == edited  # never overwritten
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "Using your edited vllm.slurm" in warnings[0] and "has changed since you copied it" in warnings[0]
    assert "Compare: diff" in warnings[0]
    _upgrade(builtin)  # another release: warned again
    resolve_template("vllm.slurm", mine)
    assert len(caplog.records) == 2


def test_copies_made_before_the_record_existed(builtin, mine, caplog):
    # a copy of an earlier released version, with no record: recognised from builtin_versions.json
    shutil.copy(builtin / "vllm.slurm", mine / "vllm.slurm")
    _upgrade(builtin)
    assert resolve_template("vllm.slurm", mine).updated
    # an edited one without a record: kept, and said to differ
    (mine / "setup_venv.sh").write_text("#!/bin/bash\n# my own setup\n")
    assert resolve_template("setup_venv.sh", mine).edited
    assert "the built-in setup_venv.sh differs from it" in caplog.text


def test_a_template_with_its_own_name_is_just_yours(builtin, mine):
    (mine / "vllm_alliance.slurm").write_text("#!/bin/bash\n")
    source = resolve_template("vllm_alliance.slurm", mine)
    assert source.yours and not source.has_builtin and source.describe().startswith("your template ")


def test_checking_does_not_update(builtin, mine):
    _copy(builtin, mine)
    old = (mine / "vllm.slurm").read_text()
    _upgrade(builtin)
    source = resolve_template("vllm.slurm", mine, update=False)
    assert source.outdated and "updated when next used" in source.describe()
    assert (mine / "vllm.slurm").read_text() == old


def test_every_built_in_template_version_is_listed():
    # builtin_versions.json must list each released version, so that unedited copies of it are
    # recognised after an upgrade. After changing a template, run scripts/update_builtin_versions.py.
    listed = json.loads((PACKAGE_DIR / "templates" / "builtin_versions.json").read_text())
    for path in (PACKAGE_DIR / "templates").iterdir():
        if path.suffix in (".slurm", ".sh"):
            assert fingerprint(path) in listed.get(path.name, []), (
                f"{path.name} changed: run scripts/update_builtin_versions.py"
            )
            assert fingerprint(path) in builtin_versions(path.name)
