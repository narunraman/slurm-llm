"""Add the current built-in templates to src/slurm_llm/templates/builtin_versions.json.

Run after changing a template, so that unedited copies of every released version are recognised
(and updated) after an upgrade. Versions already listed are kept.
"""

import hashlib
import json
from pathlib import Path

templates = Path(__file__).resolve().parent.parent / "src" / "slurm_llm" / "templates"
path = templates / "builtin_versions.json"
versions = json.loads(path.read_text())
for template in sorted(templates.iterdir()):
    if template.suffix in (".slurm", ".sh"):
        sha = hashlib.sha256(template.read_bytes()).hexdigest()
        versions[template.name] = sorted(set(versions.get(template.name, [])) | {sha})
path.write_text(json.dumps(versions, indent=2) + "\n")
print(f"updated {path}")
