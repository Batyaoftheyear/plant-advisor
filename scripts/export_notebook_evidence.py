"""Export the saved notebook tables cited by the submitted PDF report."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path


def text(value) -> str:
    return "".join(value) if isinstance(value, list) else value


class DisplayTables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.depth = 0
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.rows = []
        elif self.depth == 1 and tag == "tr":
            self.row = []
        elif self.depth == 1 and tag in {"th", "td"}:
            self.cell = []

    def handle_data(self, data):
        if self.depth == 1 and self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if self.depth == 1 and tag in {"th", "td"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif self.depth == 1 and tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None
        elif tag == "table":
            if self.depth == 1:
                self.tables.append(self.rows)
            self.depth -= 1


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    relative_source = "notebooks/03_training_and_evaluation_v3.ipynb"
    # The PDF cites this frozen notebook. The working notebook has a newer file identity.
    source_commit = "1f455d2137c01664381e594a2d1949df45720af7"
    source_bytes = subprocess.check_output(
        ["git", "show", f"{source_commit}:{relative_source}"], cwd=root,
    )
    notebook = json.loads(source_bytes)
    sft = root / "results/sft/qwen25_3b_qlora_dataset_v3"
    evaluation = root / "results/evaluation/qlora_dataset_v3"
    sft.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    records = []

    def record(path, cell_index, kind):
        content = path.read_bytes()
        records.append({"path": path.relative_to(root).as_posix(),
                        "source_cell": cell_index, "kind": kind,
                        "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})

    def tables(cell_index, names, folder):
        tables_found = []
        for output in notebook["cells"][cell_index].get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if html:
                parser = DisplayTables()
                parser.feed(text(html))
                tables_found.extend(parser.tables)
        if len(tables_found) != len(names):
            raise ValueError(f"Expected {len(names)} display tables in cell {cell_index}; found {len(tables_found)}")
        for name, rows in zip(names, tables_found):
            path = folder / name
            if rows and rows[0] and not rows[0][0]:
                rows[0][0] = "index"
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream, lineterminator="\n").writerows(rows)
            record(path, cell_index, "rendered_table; original display precision")

    def image(cell_index, name, folder):
        outputs = [o["data"]["image/png"] for o in notebook["cells"][cell_index].get("outputs", [])
                   if "image/png" in o.get("data", {})]
        if len(outputs) != 1:
            raise ValueError(f"Expected one embedded PNG in cell {cell_index}")
        path = folder / name
        path.write_bytes(base64.b64decode(text(outputs[0])))
        record(path, cell_index, "exact embedded PNG bytes")

    def streams(cell_index, name, folder):
        content = "".join(text(o["text"]) for o in notebook["cells"][cell_index].get("outputs", [])
                          if "text" in o)
        path = folder / name
        path.write_text(content, encoding="utf-8", newline="\n")
        record(path, cell_index, "saved output stream")

    tables(11, ["notebook_training_progress.csv"], sft)
    tables(13, ["notebook_training_log_excerpt.csv"], sft)
    tables(17, ["notebook_format_overall.csv", "notebook_format_by_task.csv"], evaluation)
    tables(24, ["notebook_summary.csv", "notebook_comparison_by_task.csv", "notebook_declines.csv"], evaluation)
    image(13, "loss_curves_notebook.png", sft)
    image(24, "baseline_vs_sft_notebook.png", evaluation)
    streams(11, "notebook_training_log.txt", sft)
    streams(15, "notebook_generation_log.txt", evaluation)
    streams(21, "notebook_judge_log.txt", evaluation)
    streams(23, "notebook_answer_examples.txt", evaluation)
    outputs = [o for c in notebook["cells"] for o in c.get("outputs", [])]
    manifest = {
        "origin": "saved notebook outputs cited by the submitted PDF",
        "source_notebook": relative_source,
        "source_commit": source_commit,
        "source_url": f"https://github.com/Batyaoftheyear/plant-advisor/blob/{source_commit}/{relative_source}",
        "source_notebook_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_payload_sha256": hashlib.sha256(json.dumps(outputs, ensure_ascii=False,
                                                            sort_keys=True).encode()).hexdigest(),
        "note": "CSV retains rendered display precision. Logs may be excerpts. No missing JSONL, judge explanations, or model weights were reconstructed.",
        "files": records,
    }
    (root / "results/notebook_evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"Exported {len(records)} evidence files from unchanged notebook outputs.")


if __name__ == "__main__":
    main()
