# Артефакты первого этапа

## Код, данные и результаты

| Материал | Расположение |
|---|---|
| PDF-отчёт без изменений | [report_rk1.pdf](report_rk1.pdf) |
| Исходная модель, два пробных вопроса и проверка формата | [ноутбук 01](../notebooks/01_baseline_and_judge.ipynb) |
| 30 baseline-ответов, конфигурация и график | [results/baseline/qwen25_3b_512](../results/baseline/qwen25_3b_512/) |
| Оценки baseline: 29 успешных, исходные попытки и ошибка q16 | [results/judge/qwen25_7b_baseline_v2](../results/judge/qwen25_7b_baseline_v2/) |
| Пилот PlantClinic: исходные кандидаты, источники и ошибки генерации | [pilot_generation](../results/dataset/pilot_generation/) |
| Материалы подготовки данных | [data](../data/), [карточка датасета](../DATASET_CARD.md) |
| Обучающие данные: 360 train / 40 validation, метаданные, источники | [data/v3](../data/v3/) |
| Аудит данных, 141 семейство сценариев, форма просмотра | [results/dataset/v3](../results/dataset/v3/), [ноутбук проверки](../notebooks/02_dataset.ipynb) |
| Сборка наборов | [build_sft_dataset.py](../scripts/build_sft_dataset.py), [build_sft_v3.py](../scripts/build_sft_v3.py) |
| Первый эксперимент: конфигурация, журнал до шага 135 и лучший checkpoint-45 | [results/sft/qwen25_3b_qlora_v2](../results/sft/qwen25_3b_qlora_v2/) |
| Обучение: выполненное обучение и оценка | [ноутбук 03](../notebooks/03_training_and_evaluation.ipynb) |
| Обучение: конфигурация, loss, журнал и checkpoint | [results/sft/qwen25_3b_qlora_dataset_v3_recovery](../results/sft/qwen25_3b_qlora_dataset_v3_recovery/) |
| 30 SFT-ответов, 29 оценок baseline и 30 SFT, исходные ответы судьи | [results/evaluation/qlora_dataset_v3_recovery](../results/evaluation/qlora_dataset_v3_recovery/) |
| Примеры ухудшений с объяснениями судьи | [qualitative_examples.md](../results/evaluation/qlora_dataset_v3_recovery/qualitative_examples.md) |
| Таблицы, графики и примеры, использованные в PDF | [оценка для отчёта](../results/evaluation/qlora_dataset_v3/), [обучение для отчёта](../results/sft/qwen25_3b_qlora_dataset_v3/) |
| Сверка PDF и доступных файлов | [REPORT_ARTIFACT_CHECK.md](REPORT_ARTIFACT_CHECK.md) |

## Веса

Веса, токенизаторы и конфигурации распространяются через GitHub Releases. После распаковки архивов сохраняются исходные пути внутри `results/sft/`. Ревизия базовой модели зафиксирована в `training_config.json`; инструкции загрузки — в [README_DATASET.md](../README_DATASET.md).

- [Итоговый адаптер и полный комплект результатов](https://github.com/Batyaoftheyear/plant-advisor/releases/tag/stage1-v3-reproduction). [Метаданные публикации](../results/reproduction_release.json), [манифест 45 файлов](../results/reproduction_artifacts_manifest.json), [проверка целостности](../results/reproduction_verification.json).
- [Адаптер checkpoint-45 первого эксперимента, токенизатор и журналы](https://github.com/Batyaoftheyear/plant-advisor/releases/tag/stage1-v2-checkpoint45). Оригиналы получены с Google Drive. [Метаданные публикации](../results/sft/qwen25_3b_qlora_v2/release.json), [происхождение и SHA-256](../results/sft/qwen25_3b_qlora_v2/drive_artifacts_manifest.json), [проверка](../results/sft/qwen25_3b_qlora_v2/verification.json).

У обоих адаптеров по 504 тензора F32 и 29 933 568 параметров. Проверены структура safetensors, конечность значений и хеши входных данных. Отдельная проверочная загрузка на GPU при подготовке репозитория не выполнялась.

## Происхождение таблиц PDF

В PDF применимость SFT равна 3,552, а в полном опубликованном наборе оценок — 3,586. Эти результаты не объединяются. Таблицы PDF извлечены из сохранённых выводов; они имеют точность отображения исходного ноутбука. Их происхождение, SHA-256 и ссылка на зафиксированную версию ноутбука указаны в [манифесте экспорта](../results/notebook_evidence_manifest.json).

Скрипт `scripts/export_notebook_evidence.py` читает эту версию из Git, поэтому переименование рабочего ноутбука не меняет происхождение таблиц. Полные JSONL и адаптер именно для оценки 3,552 отсутствуют. Наличие другого адаптера не устраняет это ограничение.
