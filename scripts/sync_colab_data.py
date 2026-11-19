"""Synchronize the public, checksum-pinned v3 inputs without touching v1/v2 data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_REPOSITORY = "Batyaoftheyear/plant-advisor"
MANIFEST_PATH = "data/v3/manifest.json"
SHARED_INPUTS = frozenset({
    "data/eval_questions.jsonl",
    "results/baseline/qwen25_3b_512/config.json",
    "results/baseline/qwen25_3b_512/baseline_answers.jsonl",
    "results/judge/qwen25_7b_baseline_v2/config.json",
})
REQUIRED_INPUTS = SHARED_INPUTS | {
    "data/v3/train.jsonl", "data/v3/validation.jsonl",
    "data/v3/sft_metadata.jsonl", "data/v3/sft_provenance.csv",
    "data/v3/sources.csv",
    "results/dataset/v3/v3_audit_summary.json",
    "results/dataset/v3/v3_scenario_counts.csv",
    "results/dataset/v3/manual_review_v3.csv",
}
MAX_FILE_BYTES = 16 * 1024 * 1024


def _get_bytes(url: str, limit: int = MAX_FILE_BYTES) -> bytes:
    request = Request(url, headers={"User-Agent": "PlantAdvisor-v3-Colab"})
    with urlopen(request, timeout=60) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Размер входного файла превышает допустимый лимит.")
    return content


def resolve_revision(repository: str, revision: str) -> str:
    """Resolve a branch once, so every downloaded file comes from one commit."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Ожидается публичный репозиторий в формате owner/name.")
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    # Public git refs avoid the shared unauthenticated GitHub API rate limit.
    # Disable credential helpers and prompts: Colab must not need a GitHub token.
    result = subprocess.run(
        ["git", "-c", "credential.helper=", "-c", "core.askPass=",
         "-c", "http.extraHeader=", "ls-remote", "--exit-code",
         f"https://github.com/{repository}.git", f"refs/heads/{revision}",
         f"refs/tags/{revision}", f"refs/tags/{revision}^{{}}"],
        check=True, capture_output=True, text=True, timeout=60,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""},
    )
    refs = {line.split("\t", 1)[1]: line.split("\t", 1)[0]
            for line in result.stdout.splitlines() if "\t" in line}
    commit = (refs.get(f"refs/heads/{revision}")
              or refs.get(f"refs/tags/{revision}^{{}}")
              or refs.get(f"refs/tags/{revision}"))
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("GitHub вернул некорректный идентификатор коммита.")
    return commit


def _target(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Путь файла должен быть строкой.")
    path = PurePosixPath(relative)
    if (path.is_absolute() or ".." in path.parts or "\\" in relative
            or ":" in relative or path.as_posix() != relative):
        raise ValueError(f"Недопустимый путь в манифесте: {relative}")
    allowed = relative in SHARED_INPUTS or (
        relative.startswith("data/v3/") or relative.startswith("results/dataset/v3/")
    )
    if not allowed:
        raise ValueError(f"Синхронизация не разрешает изменять этот путь: {relative}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Путь выходит за пределы каталога проекта.")
    return target


def _validate_manifest(manifest: dict) -> list[dict]:
    if manifest.get("manifest_version") != 1 or manifest.get("dataset_version") != "v3":
        raise ValueError("Ожидается манифест набора данных v3, версия формата 1.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) > 64:
        raise ValueError("Некорректный список файлов манифеста.")
    paths = []
    for item in entries:
        path = item.get("path")
        if not isinstance(path, str):
            raise ValueError("В записи манифеста отсутствует путь.")
        paths.append(path)
        if path == MANIFEST_PATH or path == "data/v3/sync_receipt.json":
            raise ValueError("Служебный файл не может входить в список данных.")
        if (type(item.get("size")) is not int
                or not 0 < item["size"] <= MAX_FILE_BYTES
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise ValueError(f"Некорректный размер или SHA-256: {path}")
    if len(paths) != len(set(paths)) or set(paths) != REQUIRED_INPUTS:
        raise ValueError("Манифест должен содержать все 12 входных файлов без повторов.")
    return entries


def _check_bytes(content: bytes, entry: dict) -> None:
    if (len(content) != entry["size"]
            or hashlib.sha256(content).hexdigest() != entry["sha256"]):
        raise ValueError(f"Не совпадает контрольная сумма: {entry['path']}")


def verify_local(project_root: str | Path) -> dict:
    """Verify every input before training; no network access and no writes."""
    root = Path(project_root).resolve()
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    for entry in _validate_manifest(manifest):
        _check_bytes(_target(root, entry["path"]).read_bytes(), entry)
    return manifest


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sync_from_github(project_root: str | Path, *, repository: str = DEFAULT_REPOSITORY,
                     revision: str = "main") -> dict:
    """Download verified v3 inputs; preserve shared experiments and back up v3 changes.

    Drive authorization belongs to google.colab.drive.mount in the notebook.
    GitHub requests are public and do not use a token or a password.
    """
    root = Path(project_root).resolve()
    commit = resolve_revision(repository, revision)
    base_url = f"https://raw.githubusercontent.com/{repository}/{commit}/"
    manifest_bytes = _get_bytes(base_url + MANIFEST_PATH, 1024 * 1024)
    manifest = json.loads(manifest_bytes)
    entries = _validate_manifest(manifest)
    # Validate every path and shared file before creating or changing anything.
    targets = {entry["path"]: _target(root, entry["path"]) for entry in entries}
    downloaded = {}
    unchanged = []
    for entry in entries:
        relative = entry["path"]
        target = targets[relative]
        if target.exists():
            current = target.read_bytes()
            if relative in SHARED_INPUTS:
                try:
                    _check_bytes(current, entry)
                except ValueError as error:
                    raise ValueError(
                        f"Общий файл {relative} отличается от GitHub. "
                        "Исходные эксперименты сохранены; проверьте версии вручную."
                    ) from error
            if (len(current) == entry["size"]
                    and hashlib.sha256(current).hexdigest() == entry["sha256"]):
                unchanged.append(relative)
                continue
        content = _get_bytes(base_url + quote(relative, safe="/"))
        _check_bytes(content, entry)
        downloaded[relative] = content

    # A failed download/hash check cannot partially replace a dataset.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    backup_root = root / "sync_backups" / "v3" / stamp
    backups = []
    pending = {**downloaded, MANIFEST_PATH: manifest_bytes}
    for relative, content in pending.items():
        target = _target(root, relative)
        if target.exists():
            if target.read_bytes() == content:
                continue
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            backups.append(relative)
        _atomic_write(target, content)

    verify_local(root)
    receipt = {
        "dataset_version": "v3", "repository": repository, "requested_revision": revision,
        "github_commit": commit, "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated": sorted(downloaded), "unchanged": sorted(unchanged),
        "backed_up": backups,
        "backup_directory": str(backup_root) if backups else None,
    }
    _atomic_write(root / "data/v3/sync_receipt.json",
                  (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--verify-local", action="store_true")
    args = parser.parse_args()
    if args.verify_local:
        verify_local(args.project_root)
        print("v3: all 12 local inputs match the manifest.")
    else:
        print(json.dumps(sync_from_github(args.project_root, repository=args.repository,
                                         revision=args.revision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
