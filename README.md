# PlantAdvisor

Учебный проект: русскоязычные консультации по уходу за комнатными растениями и разбору симптомов. Исходная `Qwen2.5-3B-Instruct` сравнивается с моделью после QLoRA на синтетическом датасете v3. Обучение и оценка выполнены на Google Colab с NVIDIA T4.

[Отчёт РК1, PDF](docs/report_rk1.pdf) · [Артефакты](docs/ARTIFACTS.md) · [Сверка с отчётом](docs/REPORT_ARTIFACT_CHECK.md) · [Инструкция запуска](README_DATASET.md)

## Результаты

| Показатель | Baseline | SFT v3 |
|---|---:|---:|
| Format Pass Rate, 30 вопросов | 73,3% | 100,0% |
| Корректность, 29 пар | 4,414 | 4,276 |
| Применимость, 29 пар | 4,000 | 3,586 |
| Диагностическое рассуждение, 14 пар | 3,357 | 3,000 |

Это показатели модели с опубликованным адаптером и полными JSONL. Структура ответа улучшилась на 26,7 процентного пункта, а содержательные оценки снизились. FPR проверяет разделы ответа и не измеряет правильность рекомендаций. Судья — `Qwen2.5-7B-Instruct`; для парного сравнения доступны 29 оценок baseline и 30 SFT. Вопрос `q20` baseline исключён после трёх неуспешных попыток получить валидный JSON.

В PDF приведены оценки другого сохранённого набора результатов: применимость 3,552 в целом и 3,286 по диагностике. Соответствующие [таблицы отчёта](results/evaluation/qlora_dataset_v3/notebook_comparison_by_task.csv) и [график](results/evaluation/qlora_dataset_v3/baseline_vs_sft_notebook.png) доступны отдельно. Их нельзя приписывать опубликованному адаптеру; полные JSONL оценок и веса именно с этими показателями не сохранились. Подробности и прочие расхождения перечислены в [сверке PDF](docs/REPORT_ARTIFACT_CHECK.md).

## Ноутбуки

| Файл | Содержание |
|---|---|
| [01_baseline_and_judge.ipynb](notebooks/01_baseline_and_judge.ipynb) | Два пробных вопроса, 30 baseline-ответов, автоматическая проверка и модель-судья |
| [02_dataset.ipynb](notebooks/02_dataset.ipynb) | Выполненный анализ данных v2 |
| [02_dataset_v3.ipynb](notebooks/02_dataset_v3.ipynb) | Проверка структуры и состава данных v3 на CPU |
| [03_training_and_evaluation_v3.ipynb](notebooks/03_training_and_evaluation_v3.ipynb) | Выполненное QLoRA-обучение, 30 SFT-ответов, оценки, таблицы и графики |

Для чтения сохранённых выводов запуск ячеек не требуется.

## Запуск в Colab

1. [Откройте ноутбук 03](https://colab.research.google.com/github/Batyaoftheyear/plant-advisor/blob/main/notebooks/03_training_and_evaluation_v3.ipynb), выберите GPU T4 и выполняйте ячейки по порядку.
2. Разрешите стандартное подключение Google Drive. Код и 12 входных файлов загружаются из одного коммита публичного GitHub с проверкой размеров и SHA-256. Токены и ручная загрузка JSONL не требуются.
3. Сохраните ZIP после обучения и полный ZIP после оценки. В них находятся адаптер, токенизатор, конфигурации, данные, логи, ответы и оценки. Локальный диск Colab временный.

[Готовые веса v3](https://github.com/Batyaoftheyear/plant-advisor/releases/tag/stage1-v3-reproduction) и [v2 checkpoint-45](https://github.com/Batyaoftheyear/plant-advisor/releases/tag/stage1-v2-checkpoint45) доступны в GitHub Releases. В архиве v3 сохранены исходные пути `results/sft/qwen25_3b_qlora_dataset_v3_recovery/` и `results/evaluation/qlora_dataset_v3_recovery/`, совпадающие с конфигурациями и выводами ноутбука. [Скачивание и загрузка адаптера](README_DATASET.md).

## Данные и метод

В v3 — 400 диалогов: 360 train и 40 validation, по 200 задач ухода и диагностики. Представлены 25 растений и 141 семейство сценариев. Семейства train (132) и validation (9) не пересекаются; 30 контрольных вопросов исключены из обучения.

QLoRA: NF4, `r=16`, `alpha=32`, dropout `0.10`, 2 эпохи, LR `3e-5`, эффективный batch size 8. Loss вычисляется только по токенам ответа. Выбор контрольной точки — по минимальному `eval_loss`; для опубликованного адаптера это шаг 90, `eval_loss=1.3437583446502686`.

[Карточка v3](docs/v3/DATASET_CARD.md) описывает состав и ограничения, [ревизия v3](docs/v3/REVISION_REPORT.md) — изменения относительно v2. Корневые `DATASET_CARD.md` и `REVISION_REPORT.md` относятся к v2. Синтетические данные и оценки судьи не прошли полную независимую предметную экспертизу.

## Структура и проверки

- `notebooks/` — код и сохранённые выводы экспериментов.
- `data/v3/` — итоговые обучающие JSONL, метаданные, источники и манифест. В `data/` сохранены предыдущие версии и контрольные вопросы.
- `results/baseline/`, `results/judge/` — исходные ответы и оценки baseline.
- `results/dataset/`, `results/sft/`, `results/evaluation/` — аудиты, обучение, ответы, оценки, таблицы и графики.
- `scripts/`, `tests/` — сборка, аудит, синхронизация, экспорт и проверки.
- `docs/` — PDF, указатель артефактов и соответствие критериям сдачи.

```bash
python scripts/audit_sft_v3.py
python scripts/sync_colab_data.py --project-root . --verify-local
python -m unittest discover -s tests -v
```

Крупные веса распространяются через Releases; локальные архивы, кеши и промежуточные checkpoints исключены из Git. Для LMS используются ссылка на репозиторий и PDF-отчёт.
