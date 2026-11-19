"""Package a v3 reproduction run, excluding optimizer states and full checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SFT_PATH = Path("results/sft/qwen25_3b_qlora_dataset_v3_recovery")
EVALUATION_PATH = Path("results/evaluation/qlora_dataset_v3_recovery")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_bundle(run_root: Path, input_root: Path, destination: Path, *,
                include_evaluation: bool = False) -> dict:
    run_root = Path(run_root).resolve()
    input_root = Path(input_root).resolve()
    destination = Path(destination).resolve()
    required = [SFT_PATH / name for name in [
        "training_config.json", "environment.json", "training_log.csv",
        "train_metrics.json", "best_checkpoint.json", "loss_curves.png",
        "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
    ]]
    if include_evaluation:
        required += [EVALUATION_PATH / name for name in [
            "sft_answers.jsonl", "sft_generation_config.json", "format_by_question.csv",
            "format_by_task.csv", "format_overall.csv", "paired_judge_scores.csv",
            "comparison_by_task.csv", "comparison_summary.json", "baseline_vs_sft.png",
        ]]
    missing = [str(path) for path in required if not (run_root / path).is_file()]
    if missing:
        raise FileNotFoundError("Отсутствуют обязательные артефакты: " + ", ".join(missing))
    training = json.loads((run_root / SFT_PATH / "training_config.json").read_text(encoding="utf-8"))
    for name in ["train", "validation"]:
        if sha256_file(input_root / f"data/v3/{name}.jsonl") != training["data_sha256"][name]:
            raise ValueError("Входные данные отличаются от конфигурации обучения.")
    selected = {}
    extensions = {".json", ".jsonl", ".csv", ".png", ".md", ".jinja", ".txt", ".safetensors"}
    for relative_folder in [SFT_PATH] + ([EVALUATION_PATH] if include_evaluation else []):
        for path in (run_root / relative_folder).rglob("*"):
            if path.is_file() and "checkpoints" not in path.relative_to(run_root).parts:
                if path.suffix in extensions:
                    selected[path.relative_to(run_root).as_posix()] = path
    manifest_path = input_root / "data/v3/manifest.json"
    inputs = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in inputs["files"]:
        path = (input_root / entry["path"]).resolve()
        if not path.is_relative_to(input_root):
            raise ValueError("Недопустимый путь входного файла.")
        if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("Не совпадает контрольная сумма входного файла: " + entry["path"])
        selected[entry["path"]] = path
    selected["data/v3/manifest.json"] = manifest_path
    artifact_manifest = {
        "run_kind": "v3_reproduction", "dataset_version": "v3",
        "stage": "complete" if include_evaluation else "training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
                  for relative, path in sorted(selected.items())],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for relative, path in sorted(selected.items()):
            archive.write(path, arcname=relative)
        archive.writestr("ARTIFACTS_MANIFEST.json",
                         json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n")
    return {"archive": str(destination), "files": len(selected),
            "bytes": destination.stat().st_size, "sha256": sha256_file(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-evaluation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(make_bundle(args.run_root, args.input_root, args.output,
                                 include_evaluation=args.include_evaluation),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
