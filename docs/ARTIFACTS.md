# Артефакты первого этапа

## Опубликованный прогон v3

Выполненный [ноутбук 04](../notebooks/04_reproduce_v3.ipynb), результаты в каталогах `*_v3_recovery` и полный архив относятся к одному запуску. Исходные ответы, оценки и файлы эксперимента сохранены без исправлений.

| Материалы | Расположение |
|---|---|
| Данные SFT: 360 train / 40 validation, источники и манифест | [data/v3](../data/v3/) |
| Аудит и форма ручного просмотра | [results/dataset/v3](../results/dataset/v3/) |
| Исходные 30 ответов и два пробных случая | [baseline](../results/baseline/qwen25_3b_512/) |
| Автоматическая проверка и примеры её срабатывания | [ноутбук 01](../notebooks/01_baseline_and_judge.ipynb), раздел 4 |
| Все 30 ответов дообученной модели | [sft_answers.jsonl](../results/evaluation/qlora_dataset_v3_recovery/sft_answers.jsonl) |
| 29 оценок baseline, 30 SFT, исходные попытки и ошибка q20 | [judge_b2a7a9c6e1084a42](../results/evaluation/qlora_dataset_v3_recovery/judge_b2a7a9c6e1084a42/) |
| Парные оценки, агрегаты и график сравнения | [results/evaluation/qlora_dataset_v3_recovery](../results/evaluation/qlora_dataset_v3_recovery/) |
| Примеры ухудшений с ответами и объяснениями судьи | [qualitative_examples.md](../results/evaluation/qlora_dataset_v3_recovery/qualitative_examples.md) |
| Настройки, окружение, журнал, loss и выбранный checkpoint | [results/sft/qwen25_3b_qlora_dataset_v3_recovery](../results/sft/qwen25_3b_qlora_dataset_v3_recovery/) |
| Веса safetensors, конфигурация и полный токенизатор | [GitHub Release](https://github.com/Batyaoftheyear/plant-advisor/releases/tag/stage1-v3-reproduction), ZIP и `SHA256SUMS.txt` |
| Происхождение и контрольные суммы 45 файлов архива | [манифест](../results/reproduction_artifacts_manifest.json) |
| Результаты проверки целостности и метрик | [reproduction_verification.json](../results/reproduction_verification.json) |
| Адреса опубликованного архива и его SHA-256 | [reproduction_release.json](../results/reproduction_release.json) |

В ZIP сохранена структура репозитория. Адаптер расположен в `results/sft/qwen25_3b_qlora_dataset_v3_recovery/adapter/adapter_model.safetensors`. Его размер — 119 801 528 байт; 504 тензора содержат 29 933 568 параметров. Веса и полный токенизатор распространяются через Releases из-за размера. [Скачивание и запуск](../README_DATASET.md).

Проверены CRC архива, SHA-256 всех 45 файлов, структура safetensors, конечность значений, согласованность входных данных и конфигураций. Метрики пересчитаны из отдельных оценок. Все 59 успешных оценок сопоставлены с вопросами, ответами и исходными ответами судьи. Независимая загрузка адаптера на GPU при подготовке репозитория не выполнялась; фактическое обучение и оценка сохранены в ноутбуке 04.

## Предыдущие эксперименты

Данные v1/v2, [выполненный анализ v2](../notebooks/02_dataset.ipynb) и [первый прогон обучения v3](../notebooks/03_training_and_evaluation_v3.ipynb) сохранены. Первый отдельный судья находится в [results/judge/qwen25_7b_baseline_v2](../results/judge/qwen25_7b_baseline_v2/): 29 оценок, пропуск q16. Он отличается от судьи парного сравнения, где пропущен q20 baseline.

Веса первого обучения v3 не сохранились. Его таблицы, графики и текстовые выводы экспортированы скриптом `scripts/export_notebook_evidence.py`; [манифест экспорта](../results/notebook_evidence_manifest.json) содержит хеши исходного ноутбука, всех выводов и экспортированных файлов. CSV сохраняют точность отображения; журнал является выдержкой, примеры — выбранным поднабором.

Применимость SFT первого запуска равна 3,552, повторного опубликованного — 3,586. Числа разных запусков не объединяются. Все сохранённые выводы ноутбуков, включая предупреждения и ошибки интерактивных сеансов, оставлены без изменения.
