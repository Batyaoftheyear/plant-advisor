# Артефакты эксперимента

| Материал | Файлы |
|---|---|
| Отчёт | [report_rk1.pdf](report_rk1.pdf) |
| Исходная модель и модель-судья | [ноутбук 01](../notebooks/01_baseline_and_judge.ipynb) |
| Подготовка данных | [ноутбук 02](../notebooks/02_dataset.ipynb), [сборщик](../scripts/build_sft_v3.py), [карточка датасета](../DATASET_CARD.md) |
| Обучение и итоговое сравнение | [ноутбук 03](../notebooks/03_training_and_evaluation.ipynb) |
| Контрольные вопросы | [eval_questions.jsonl](../data/eval_questions.jsonl) |
| Обучающие JSONL, метаданные и источники | [data/v3](../data/v3/) |
| Проверки данных и форма ручного просмотра | [results/dataset/v3](../results/dataset/v3/) |
| Пилот генерации по внешнему корпусу | [pilot_generation](../results/dataset/pilot_generation/) |
| 30 ответов baseline, настройки и FPR | [results/baseline/qwen25_3b_512](../results/baseline/qwen25_3b_512/) |
| Начальная оценка baseline: 29 результатов, пропуск q16 | [results/judge/qwen25_7b_baseline_v2](../results/judge/qwen25_7b_baseline_v2/) |
| Первый эксперимент: конфигурация, журнал и checkpoint-45 | [results/sft/qwen25_3b_qlora_v2](../results/sft/qwen25_3b_qlora_v2/) |
| Итоговое обучение: конфигурация, loss и checkpoint | [results/sft/qwen25_3b_qlora_dataset_v3_recovery](../results/sft/qwen25_3b_qlora_dataset_v3_recovery/) |
| 30 SFT-ответов, 29 оценок baseline, 30 оценок SFT и исходные ответы судьи | [results/evaluation/qlora_dataset_v3_recovery](../results/evaluation/qlora_dataset_v3_recovery/) |
| Примеры ухудшений с объяснениями судьи | [qualitative_examples.md](../results/evaluation/qlora_dataset_v3_recovery/qualitative_examples.md) |

## Веса

- [Итоговый адаптер и полный комплект результатов](https://github.com/Batyaoftheyear/plant-advisor/releases/tag/stage1-v3-reproduction). [Адрес архива и SHA-256](../results/reproduction_release.json), [манифест файлов](../results/reproduction_artifacts_manifest.json).
- [Адаптер checkpoint-45 первого эксперимента](https://github.com/Batyaoftheyear/plant-advisor/releases/tag/stage1-v2-checkpoint45). [Адрес архива и SHA-256](../results/sft/qwen25_3b_qlora_v2/release.json), [происхождение файлов](../results/sft/qwen25_3b_qlora_v2/drive_artifacts_manifest.json).

В архивах находятся адаптеры safetensors, конфигурации и токенизаторы. Базовая модель — `Qwen/Qwen2.5-3B-Instruct`; её ревизия указана в `training_config.json`. У каждого адаптера 504 тензора F32 и 29 933 568 параметров. [Инструкция загрузки и запуска](../README_DATASET.md).

## Наборы оценки

| ID оценки | Применимость SFT, 29 пар | Применимость по диагностике, 14 пар | Доступные материалы |
|---|---:|---:|---|
| `b2a7a9c6e1084a42` | 3,586 | 3,357 | [Полные JSONL, парные оценки и графики](../results/evaluation/qlora_dataset_v3_recovery/); адаптер в Releases |
| `d88b49509b9a8ef1` | 3,552 | 3,286 | [Таблицы, графики и примеры](../results/evaluation/qlora_dataset_v3/); веса и полные JSONL недоступны |

Оценки относятся к разным экспериментальным записям. Средние из разных записей не объединяются. Во второй строке таблицы значения извлечены из отображённых выводов ноутбука и имеют точность отображения; журналы являются выдержками, примеры — поднабором. Источник и SHA-256 указаны в [манифесте](../results/notebook_evidence_manifest.json).
