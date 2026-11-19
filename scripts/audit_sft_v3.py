"""Read-only structural and cross-file audit of the isolated v3 dataset."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def audit(root: Path) -> dict:
    data = root / "data/v3"
    metadata = read_jsonl(data / "sft_metadata.jsonl")
    train = read_jsonl(data / "train.jsonl")
    validation = read_jsonl(data / "validation.jsonl")
    evaluation = read_jsonl(root / "data/eval_questions.jsonl")
    assert (len(metadata), len(train), len(validation), len(evaluation)) == (400, 360, 40, 30)
    assert len({r["id"] for r in metadata}) == 400
    assert all(r["id"].startswith("SFT3-") for r in metadata)
    normalize = lambda text: " ".join(re.findall(r"\w+", text.casefold()))
    assert len({normalize(r["question"]) for r in metadata}) == 400
    assert len({normalize(r["answer"]) for r in metadata}) == 400
    assert not ({normalize(r["question"]) for r in metadata}
                & {normalize(r["question"]) for r in evaluation})
    assert len({r["plant"] for r in metadata}) == 25
    assert set(Counter(r["plant"] for r in metadata).values()) == {16}
    with (data / "sources.csv").open(encoding="utf-8-sig", newline="") as stream:
        sources = {row["source_id"]: row for row in csv.DictReader(stream)}
    with (data / "sft_provenance.csv").open(encoding="utf-8-sig", newline="") as stream:
        provenance_rows = list(csv.DictReader(stream))
    provenance = {r["id"]: r for r in provenance_rows}
    assert len(provenance_rows) == len(provenance) == 400
    assert set(provenance) == {r["id"] for r in metadata}
    families = {}
    for split, rows in [("train", train), ("validation", validation)]:
        subset = [r for r in metadata if r["split"] == split]
        assert len(subset) == len(rows)
        assert Counter(r["task"] for r in subset) == {
            "care": len(rows) // 2, "diagnosis": len(rows) // 2,
        }
        families[split] = {(r["task"], r["scenario"]) for r in subset}
        for record, row in zip(subset, rows):
            assert row == {"messages": [
                {"role": "user", "content": record["question"]},
                {"role": "assistant", "content": record["answer"]},
            ]}
            headings = (["Условия", "Рекомендации"] if record["task"] == "care"
                        else ["Возможные причины", "Что проверить", "Что делать"])
            matches = [list(re.finditer(rf"(?m)^\s*{re.escape(h)}:\s*$",
                                       record["answer"])) for h in headings]
            assert all(len(group) == 1 for group in matches)
            positions = [group[0].start() for group in matches]
            assert positions == sorted(positions)
            for i, group in enumerate(matches):
                end = positions[i + 1] if i + 1 < len(matches) else len(record["answer"])
                assert record["answer"][group[0].end():end].strip()
            assert [sources[s]["url"] for s in record["source_ids"]] == record["source_urls"]
            for field in ["split", "task", "plant", "scenario", "generation_method",
                          "evidence_level", "expert_review"]:
                assert provenance[record["id"]][field] == record[field]
    assert not families["train"] & families["validation"]
    assert (len(families["train"]), len(families["validation"])) == (132, 9)
    report_dir = root / "results/dataset/v3"
    summary = json.loads((report_dir / "v3_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_version"] == "v3" and summary["total_examples"] == 400
    assert summary["scenario_families_total"] == 141
    assert summary["train_validation_scenario_overlap_count"] == 0
    with (report_dir / "v3_scenario_counts.csv").open(encoding="utf-8-sig", newline="") as stream:
        counts = list(csv.DictReader(stream))
    assert len(counts) == 141
    assert sum(int(r["train"]) for r in counts) == 360
    assert sum(int(r["validation"]) for r in counts) == 40
    with (report_dir / "manual_review_v3.csv").open(encoding="utf-8-sig", newline="") as stream:
        review = list(csv.DictReader(stream))
    assert len(review) == 30 and len({r["id"] for r in review}) == 30
    assert {r["id"] for r in review} <= set(provenance)
    return {"dataset_version": "v3", "train": 360, "validation": 40,
            "scenario_families": 141, "scenario_overlap": 0, "manual_review": 30}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
