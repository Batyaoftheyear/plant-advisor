"""Reproducible second-pass revision of PlantAdvisor's synthetic SFT dataset.

Run after build_sft_dataset.py. Uses the original metadata and preserves record IDs,
train/validation assignments, and the messages schema. No model calls needed.
"""
from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "results" / "dataset"
BUILD_SCRIPT = ROOT / "scripts" / "build_sft_dataset.py"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_plant_data() -> dict[str, dict[str, str]]:
    """Extract only literal plant reference table, without executing the generator."""
    tree = ast.parse(BUILD_SCRIPT.read_text(encoding="utf-8"))
    table = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PLANTS" for t in node.targets)
    )
    return {
        x[0]: {
            "name": x[0], "gen": x[1], "dat": x[2], "latin": x[3],
            "group": x[4], "light": x[5], "water": x[6], "humidity": x[7],
            "root": x[8], "special": x[9], "plant_source": x[10],
        }
        for x in table
    }


def put_in_section(answer: str, heading: str, text: str) -> str:
    """Append one relevant recommendation to an existing answer section."""
    parts = answer.split(f"{heading}:\n")
    assert len(parts) == 2, heading
    prefix, suffix = parts
    other_heading = re.search(r"\n\n(?:Условия|Рекомендации|Возможные причины|Что проверить|Что делать):\n", suffix)
    if other_heading:
        body, rest = suffix[:other_heading.start()], suffix[other_heading.start():]
    else:
        body, rest = suffix, ""
    number = max([int(v) for v in re.findall(r"(?m)^(\d+)\. ", body)] or [0])
    extra = f"{number + 1}. {text}" if number else f"- {text}"
    return prefix + f"{heading}:\n" + body.rstrip() + "\n" + extra + rest


# Check-list prompts specific to the plant's anatomy, instead of general
# leaf/petiole wording applied to all houseplants indiscriminately.
DETAIL_BY_GROUP = {
    "orchid": "Осмотрите также основание розетки и воздушные корни; состояние коры в горшке оценивайте отдельно от влажности воздуха.",
    "succulent": "Оцените упругость запасающих воду тканей и убедитесь, что субстрат успевает просохнуть между поливами.",
    "ficus": "Уточните, не меняли ли недавно место горшка, не было ли холодного сквозняка или резкого изменения освещения.",
    "humid": "Сравните влажность воздуха и субстрата: сухой воздух не означает, что корням нужна дополнительная вода.",
    "moist": "Проверьте и влажность субстрата, и качество освещения: растение чувствительно к длительной пересушке и сырости.",
    "tropical": "Проверьте вентиляцию и влажность субстрата отдельно: слабое освещение увеличивает риск перелива.",
    "semi": "Проверьте, нет ли длительной сырости в центре кома при сухой поверхности грунта.",
    "sunny": "Сопоставьте симптомы с прямым солнцем и временем последней перестановки горшка.",
    "cool": "Уточните, не находится ли горшок рядом с отопительным прибором: перегрев и сухой воздух влияют на состояние растения.",
}


def species_check(p: dict, scenario: str) -> str:
    """Generate a focused species-level check using explicit plant care facts."""
    n = p["gen"]
    if scenario == "leaf_fall":
        return f"Для {n} сопоставьте условия места с рекомендуемым освещением: {p['light']}."
    if scenario in {"gnats", "soft_base"}:
        return f"При оценке риска сырости у {n} учитывайте нормальный режим: {p['water']}."
    if scenario == "brown_spots":
        return f"У {n} отдельно проверьте полив: {p['water']}; по одному цвету пятен нельзя определить причину."
    if scenario == "feeding":
        return f"Для {n} ориентируйтесь на доступный свет: {p['light']}; при недостатке света подкормка не решит проблему роста."
    if scenario == "mealybugs":
        return f"У растения «{p['name']}» осмотрите места соединения листьев и побегов, молодые части и труднодоступные участки; оцените, где расположены колонии."
    if scenario == "mites":
        return f"Для {n} отметьте, на каких листьях и побегах появились точки, и повторите осмотр через несколько дней, чтобы оценить распространение."
    if scenario == "sunburn":
        return f"У растения «{p['name']}» проверьте соответствие освещения потребностям вида: {p['light']}."
    return DETAIL_BY_GROUP[p['group']]


CACTUS_QUESTIONS = {
    "yellowing": "У пустынного кактуса часть стебля желтеет. Что проверить прежде всего?",
    "crispy_tips": "У пустынного кактуса верхушка стебля подсыхает. Как выяснить причину?",
    "leaf_fall": "У пустынного кактуса отсыхают и отделяются боковые отростки. Что проверить?",
    "pale_growth": "У пустынного кактуса новые приросты вытянутые и бледные. Что проверить?",
    "brown_spots": "На стебле пустынного кактуса появились бурые пятна. Можно ли определить причину без осмотра?",
    "curl": "Стебель пустынного кактуса заметно сморщился. Как различить пересушку и повреждение корней?",
    "mites": "На стебле пустынного кактуса видны светлые точки и тонкая паутинка. Что проверить?",
    "soft_base": "Основание пустынного кактуса размягчилось, а субстрат мокрый. Что делать?",
    "sunburn": "После перестановки на солнечное окно на стебле кактуса появились светлые сухие участки. Что проверить?",
    "wilt": "Стебель пустынного кактуса потерял упругость. Как понять, хватает ли влаги корням?",
}

CACTUS_ANSWERS = {
    "yellowing": (
        ["Пожелтение стебля может быть связано со стрессом от избытка влаги, резкой смены освещения или повреждением тканей.", "По одному изменению окраски нельзя утверждать, что это грибковое заболевание."],
        ["Проверьте, сухой ли субстрат в глубине, есть ли дренажные отверстия и не холодно ли у окна.", "Уточните, мягкая ли пожелтевшая ткань, увеличивается ли зона и не появились ли мокнущие участки."],
        ["Если основание мягкое и грунт мокрый, прекратите полив до осмотра корней и состояния стебля.", "Если стебель плотный и повреждение не растёт, стабилизируйте условия и отслеживайте изменения."]
    ),
    "crispy_tips": (
        ["Сухой участок на верхушке бывает после ожога, механического повреждения или длительного стресса; иногда требуется исключить поражение тканей.", "Точную причину нельзя установить без оценки состояния участка."],
        ["Выясните, сухой и твёрдый участок или влажный и мягкий; осмотрите границу повреждения.", "Проверьте недавнюю смену освещения, температуру, влажность субстрата и дренаж."],
        ["Если участок сухой и не растёт, наблюдайте за ним и избегайте резкого прямого солнца.", "При увеличении мягкой или мокнущей зоны изолируйте растение и проверьте состояние корней и стебля."]
    ),
    "leaf_fall": (
        ["Усыхание отростков может быть связано с повреждением, недостатком света или проблемами корней; причина зависит от вида кактуса.", "Не все пустынные кактусы образуют боковые отростки; важно выяснить строение растения."],
        ["Определите вид и проверьте, отростки отпадают сухими или у основания есть мягкая ткань.", "Оцените освещение, глубинную влажность смеси и состояние основания стебля."],
        ["При мягких и мокнущих участках временно прекратите полив и осмотрите растение на гниль.", "При сухом повреждении стабилизируйте условия и не пересаживайте без признаков проблем с корнями."]
    ),
    "pale_growth": (
        ["Бледные, тонкие и вытянутые новые приросты часто связаны с недостатком света; это признак этиоляции.", "Сильная подкормка не заменит необходимое освещение."],
        ["Оцените реальный свет у кактуса и сравните новый прирост со старым.", "Проверьте, не поддерживается ли зимой слишком тёплый режим с частым поливом."],
        ["Постепенно увеличьте освещённость до подходящей виду, избегая внезапного ожога.", "Не учащайте полив и подкормку для исправления вытянувшихся частей; здоровый новый рост будет отличаться по форме."]
    ),
    "brown_spots": (
        ["Бурые пятна на стебле могут быть сухими рубцами, ожогом или признаком повреждения тканей и корней.", "Без осмотра влажности и плотности участка нельзя назвать конкретную болезнь."],
        ["Уточните, пятна сухие, твёрдые или мокнущие и мягкие; появляются ли новые участки.", "Проверьте освещение, состояние основания, глубину просушки и дренаж."],
        ["При сухом стабильном рубце наблюдайте без агрессивной обработки.", "При быстро распространяющемся размягчении изолируйте растение и осмотрите ткани; не увлажняйте мокрый грунт дополнительно."]
    ),
    "curl": (
        ["Сморщивание стебля бывает при обезвоживании, но возможно и при повреждении корней в мокром субстрате.", "Одного внешнего вида недостаточно, чтобы определить необходимость полива."],
        ["Проверьте глубинную влажность смеси и отсутствие воды в кашпо.", "Осмотрите основание стебля на размягчение, изменение окраски и запах; учитывайте сезон и температуру."],
        ["При сухом субстрате и здоровом плотном основании поливайте с учётом потребности вида и сезона.", "При сыром грунте не добавляйте воду, пока не проверите дренаж и состояние корней."]
    ),
    "mites": (
        ["Точечное обесцвечивание и тонкая паутина могут сопровождать поражение паутинными клещами.", "Не каждая паутинка доказывает наличие клещей без их обнаружения."],
        ["С лупой осмотрите поверхность стебля и ареолы; ищите подвижных вредителей, а не только следы повреждения.", "Проверьте соседние растения и новые повреждения."],
        ["Изолируйте кактус и аккуратно удалите подтверждённых вредителей способом, который не повреждает ткани и колючки.", "При выраженном заражении используйте только препарат, разрешённый для комнатных кактусов и конкретного вредителя, по инструкции."]
    ),
    "soft_base": (
        ["Мягкое основание при мокром грунте может указывать на повреждение стебля или корней из-за переувлажнения.", "Определять возбудителя по описанию без осмотра нельзя."],
        ["Проверьте, насколько долго субстрат сырой, есть ли дренаж и мокнущие участки на основании.", "Осмотрите корни и стебель на запах и размягчение при ухудшении состояния."],
        ["Не поливайте дальше до выяснения причины; изолируйте растение при признаках гнили.", "Если обнаружены нежизнеспособные ткани, потребуется санитарная обработка и хорошо дренируемая смесь; действуйте осторожно."]
    ),
    "wilt": (
        ["Стебель может терять упругость как при длительной пересушке, так и при повреждении корней в переувлажнённой смеси.", "Без проверки кома и основания кактуса нельзя рекомендовать немедленный полив."],
        ["Оцените глубинную влажность субстрата, наличие воды в кашпо и температуру содержания.", "Осмотрите стебель у основания: нет ли мягких участков, потемнения или неприятного запаха."],
        ["Если смесь сухая и ткани упругие у основания, полейте с учётом времени года и потребности вида.", "Если субстрат мокрый или основание мягкое, воду не добавляйте, проверьте дренаж и состояние корней."]
    ),
    "sunburn": (
        ["Светлые сухие пятна после внезапного сильного солнца могут быть ожогом стебля.", "Если участок мягкий или мокнет, нужно исключить другую проблему."],
        ["Установите, на стороне ли прямого солнца возникло повреждение и как быстро оно появилось.", "Проверьте плотность тканей и увеличение площади пятна."],
        ["Верните более мягкое освещение и повышайте его постепенно.", "Не компенсируйте ожог лишним поливом; повреждённая ткань может остаться рубцовой."]
    ),
}


def cactus_answer(scenario: str) -> str:
    causes, checks, actions = CACTUS_ANSWERS[scenario]
    return (
        "Возможные причины:\n" + "\n".join("- " + s for s in causes)
        + "\n\nЧто проверить:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(checks, 1))
        + "\n\nЧто делать:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(actions, 1))
    )


def make_revisions(row: dict, plants: dict) -> tuple[dict, list[str]]:
    new = row.copy()
    plant = plants[row["plant"]]
    scenario = row["scenario"]
    reasons = []
    q = new["question"]
    a = new["answer"]

    # Wrong case after 'на': several prompts used the dative instead of a location form.
    if q.startswith("После перестановки к яркому окну на "):
        q = f"После перестановки к яркому окну на листьях {plant['gen']} появились светлые сухие пятна. Что проверить?"
        reasons.append("grammar: corrected prepositional case")

    if q.startswith("Что изменить в содержании "):
        q = f"Зимой света становится меньше. Как скорректировать условия для растения «{plant['name']}»?"
        reasons.append("wording: unnatural phrasing replaced")

    if q.startswith("Как защитить ") and " от сухого воздуха " in q:
        q = f"Как снизить влияние сухого воздуха от отопления на растение «{plant['name']}»?"
        reasons.append("grammar: corrected case in humidity question")

    if q.startswith("По каким признакам понять, что ") and q.endswith(" нужна пересадка?"):
        q = f"По каким признакам понять, что пора пересаживать растение «{plant['name']}»?"
        reasons.append("grammar: corrected case in repotting question")

    if q.startswith("На ") and " появились мелкие светлые точки и паутинка." in q:
        q = f"На растении «{plant['name']}» появились мелкие светлые точки и паутинка. Может ли это быть вредитель?"
        reasons.append("grammar: corrected case in mite question")

    if q.startswith("Белые ватные комочки на "):
        q = f"На растении «{plant['name']}» появились белые ватные комочки. Как отличить вредителей от налёта?"
        reasons.append("grammar: corrected case in mealybug question")

    if q.endswith(" выглядит вялым. Как отличить недостаток воды от проблем с корнями?"):
        q = f"Растение «{plant['name']}» выглядит вялым. Как отличить пересушку от проблем с корнями?"
        reasons.append("wording: standardized noun phrase")

    if row["plant"] == "пустынный кактус":
        if scenario in CACTUS_QUESTIONS:
            q = CACTUS_QUESTIONS[scenario]
            a = cactus_answer(scenario)
            reasons.append("botanical: replaced leaf-based cactus scenario with stem/anatomy-appropriate case")
        elif scenario == "humidity":
            q = "Нужно ли специально повышать влажность воздуха для пустынного кактуса зимой?"
            a = ("Условия:\n- Большинство пустынных кактусов нормально переносит сухой воздух квартиры; влажный воздух не заменяет свет и правильный режим полива.\n"
                 "- Зимой особенно нежелательно сочетать холодный и сырой субстрат.\n\nРекомендации:\n"
                 "1. Не устанавливайте увлажнитель специально для пустынного кактуса без видовой необходимости.\n"
                 "2. Держите горшок подальше от прямого горячего потока батареи и обеспечьте подходящий свет.\n"
                 "3. Проверяйте просыхание субстрата и сократите полив при прохладной зимовке.")
            reasons.append("botanical: cactus humidity care made species-appropriate")

    # Add one relevant species-specific recommendation to scenarios originally identical for 10-20 plant species.
    if scenario == "feeding":
        a = put_in_section(a, "Условия", "Для этого растения важно подходящее освещение: " + plant["light"] + ".")
        reasons.append("specificity: feeding and light requirements")
    elif scenario in {"leaf_fall", "mites", "mealybugs", "gnats", "brown_spots", "soft_base", "sunburn"} and not (row["plant"] == "пустынный кактус" and scenario in CACTUS_ANSWERS):
        a = put_in_section(a, "Что проверить", species_check(plant, scenario))
        reasons.append("specificity: plant-specific diagnostic check")

    # One redundant sunlight text usually inherited from leaf-based templates; correct species coverage.
    if row["plant"] == "фаленопсис" and scenario in {"wilt", "yellowing", "gnats", "soft_base"}:
        a = a.replace("грунт", "субстрат")
        reasons.append("botanical: orchid substrate terminology")

    # Avoid claiming that the linked general references independently validate each record.
    new["question"] = q
    new["answer"] = a
    eval_rows = read_jsonl(DATA / "eval_questions.jsonl")
    normalize = lambda txt: " ".join(re.findall(r"\w+", txt.lower(), flags=re.UNICODE))
    sim, closest = max(
        (SequenceMatcher(None, normalize(q), normalize(e["question"])).ratio(), e["id"])
        for e in eval_rows
    )
    new["highest_eval_similarity"] = round(sim, 3)
    new["closest_eval_id"] = closest
    new["quality_note"] = "synthetic_general_references;automated_checks;human_expert_review_not_completed"
    new["revision_status"] = "revised" if reasons else "unchanged_after_screening"
    new["revision_reasons"] = reasons
    return new, reasons


def validate(records: list[dict]) -> dict:
    assert len(records) == 400
    ids = [r["id"] for r in records]
    assert len(set(ids)) == 400
    q = [r["question"].strip() for r in records]
    a = [r["answer"].strip() for r in records]
    assert len(set(q)) == len(q), "Duplicate questions"
    assert len(set(a)) == len(a), "Duplicate answers"
    assert all(r["task"] in {"care", "diagnosis"} for r in records)
    assert all(len(r["source_urls"]) > 0 for r in records)
    for r in records:
        hdrs = ["Условия:", "Рекомендации:"] if r["task"] == "care" else ["Возможные причины:", "Что проверить:", "Что делать:"]
        starts = [r["answer"].find(h) for h in hdrs]
        assert all(n >= 0 for n in starts) and starts == sorted(starts), r["id"]
        assert not re.search(r"[一-龯]|iteDatabase|select \* from", r["answer"], flags=re.I)
        if r["plant"] == "пустынный кактус":
            assert not re.search(r"лист[ьяуе]|листопад", r["question"].lower()), (r["id"], r["question"])
    norm = lambda s: " ".join(re.findall(r"\w+", s.lower(), flags=re.UNICODE))
    eval_q = {norm(r["question"]) for r in read_jsonl(DATA / "eval_questions.jsonl")}
    assert not (set(map(norm, q)) & eval_q)
    cnt = Counter((r["split"], r["task"]) for r in records)
    assert cnt == {("train", "care"): 180, ("train", "diagnosis"): 180, ("validation", "care"): 20, ("validation", "diagnosis"): 20}
    return {"total": 400, "train": 360, "validation": 40, "unique_questions": len(set(q)), "unique_answers": len(set(a)), "cactus_question_leaf_errors": 0, "exact_eval_overlap": 0}


def write_outputs(original: list[dict], revised: list[dict], change_log: list[dict], stats: dict) -> None:
    by_id = {r["id"]: r for r in revised}
    save_jsonl(DATA / "sft_metadata.jsonl", revised)
    for split, fn in [("train", "train.jsonl"), ("validation", "validation.jsonl")]:
        save_jsonl(DATA / fn, [
            {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}]}
            for r in revised if r["split"] == split
        ])

    original_prov = list(csv.DictReader((DATA / "sft_provenance.csv").open(encoding="utf-8-sig")))
    fieldnames = list(dict.fromkeys(list(original_prov[0]) + ["revision_status", "revision_reasons"]))
    with (DATA / "sft_provenance.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in original_prov:
            rec = by_id[row["id"]]
            row.update(revision_status=rec["revision_status"], revision_reasons=" | ".join(rec["revision_reasons"]),
                       quality_note=rec["quality_note"],
                       highest_eval_similarity=rec["highest_eval_similarity"], closest_eval_id=rec["closest_eval_id"])
            writer.writerow(row)

    with (REPORTS / "revision_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "plant", "task", "scenario", "change_type", "old_question", "new_question", "old_answer", "new_answer"])
        writer.writeheader()
        writer.writerows(change_log)

    with (REPORTS / "revision_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Human review must be re-done after changes; select a stable and diversified 30-record sample.
    selected = sorted(revised, key=lambda r: r["id"])
    selected_ids = {f"SFT-{i:04d}" for i in [2, 21, 23, 87, 114, 117, 122, 123, 124, 125, 126, 127, 128, 135, 161, 166, 179, 201, 215, 237, 260, 283, 301, 325, 342, 363, 372, 389, 395, 399]}
    # The file is a worksheet, not an assertion that experts have already reviewed these rows.
    with (REPORTS / "manual_review_sample.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "task", "plant", "scenario", "question", "answer", "source_urls", "factually_correct", "actionable", "reviewer_notes"])
        w.writeheader()
        for row in selected:
            if row["id"] in selected_ids:
                w.writerow({k: (" | ".join(row[k]) if k == "source_urls" else row.get(k, "")) for k in w.fieldnames})


if __name__ == "__main__":
    original = read_jsonl(DATA / "sft_metadata_raw.jsonl")
    plants = get_plant_data()
    revised = []
    changes = []
    for rec in original:
        new, reasons = make_revisions(rec, plants)
        revised.append(new)
        if reasons:
            changes.append({
                "id": rec["id"], "plant": rec["plant"], "task": rec["task"],
                "scenario": rec["scenario"], "change_type": " | ".join(reasons),
                "old_question": rec["question"], "new_question": new["question"],
                "old_answer": rec["answer"], "new_answer": new["answer"],
            })
    stats = validate(revised)
    stats.update({
        "original_exact_duplicate_answers": len(original) - len({r["answer"] for r in original}),
        "corrected_records": len(changes),
        "unchanged_records": len(revised) - len(changes),
        "known_limitations": [
            "Automated and targeted substantive revision, not 400 independent expert fact checks",
            "References document general plant-care principles; not every generated sentence is individually source-verified",
            "Many examples retain repeated structural templates; zero exact duplicate answers does not mean zero semantic duplication",
            "Manual review worksheet is not completed",
        ],
    })
    write_outputs(original, revised, changes, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
