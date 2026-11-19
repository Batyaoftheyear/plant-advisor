"""Rebuild the public Colab input manifest after an intentional v3 data change."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sync_colab_data import DEFAULT_REPOSITORY, REQUIRED_INPUTS, verify_local


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = []
    for relative in sorted(REQUIRED_INPUTS):
        data = (root / relative).read_bytes()
        files.append({"path": relative, "size": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"manifest_version": 1, "dataset_version": "v3",
                "repository": DEFAULT_REPOSITORY, "files": files}
    (root / "data/v3/manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    verify_local(root)
    print("Updated and verified the v3 manifest (12 inputs).")


if __name__ == "__main__":
    main()
