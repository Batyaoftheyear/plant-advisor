# Запуск PlantAdvisor v3

## Готовые результаты и веса

[Ноутбук 03](notebooks/03_training_and_evaluation_v3.ipynb) содержит выполненное обучение на T4, 30 ответов SFT и парное сравнение с baseline. Все 59 выводов ячеек сохранены.

В [GitHub Release](https://github.com/Batyaoftheyear/plant-advisor/releases/tag/stage1-v3-reproduction) доступны `plant_advisor_v3_reproduction_complete.zip` и `SHA256SUMS.txt`. Архив содержит адаптер safetensors, токенизатор, данные, конфигурации, логи, ответы и оценки. Размер ZIP — 115 469 576 байт. [Метаданные публикации](results/reproduction_release.json).

Проверьте SHA-256 и распакуйте ZIP в корень копии репозитория, сохраняя пути:

```text
9ed68385ac0b0ac3025c5f78f825354d31117fff6d0479fb1ec6eb471ebc0443
```

Адаптер находится в `results/sft/qwen25_3b_qlora_dataset_v3_recovery/adapter/`. Для загрузки базовой модели используйте `base_model` и `base_revision` из соседнего `training_config.json`: `Qwen/Qwen2.5-3B-Instruct`, ревизия `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Токенизатор загружайте из каталога адаптера. Поле `revision` в автоматически созданном `adapter_config.json` равно `null`, поэтому ревизию базы необходимо указать явно. Версии библиотек и способ 4-битной загрузки приведены в ноутбуке 03.

## Обучение и оценка в Colab

[Открыть ноутбук 03 в Google Colab](https://colab.research.google.com/github/Batyaoftheyear/plant-advisor/blob/main/notebooks/03_training_and_evaluation_v3.ipynb).

1. Выберите T4 и выполните ячейки по порядку. Если установка библиотек потребует перезапуска, перезапустите среду и начните выполнение заново.
2. В окне `drive.mount` разрешите подключение Google Drive. Публичные GitHub-данные и базовая модель доступны без токенов. Можно использовать пустой Google Drive: входные файлы загрузятся автоматически.
3. Сохраните `plant_advisor_v3_reproduction_training.zip`, скачиваемый после обучения. В нём находятся адаптер, конфигурация и журналы.
4. После оценки сохраните `plant_advisor_v3_reproduction_complete.zip`: он дополнительно содержит ответы, оценки судьи, таблицы и графики.

Результаты записываются в `/content/plant-advisor-v3-reproduction/`, в подкаталоги `results/sft/qwen25_3b_qlora_dataset_v3_recovery/` и `results/evaluation/qlora_dataset_v3_recovery/`. Локальный диск Colab временный: архивы необходимо сохранить до завершения сеанса.

Экспорт выполняется `scripts/export_submission_artifacts.py`. Перед упаковкой проверяются адаптер и согласованность входных данных с конфигурацией. В ZIP включаются контрольные суммы; состояния оптимизатора и промежуточные checkpoints не включаются.

## Данные и аудит

Синхронизация использует один SHA коммита и `data/v3/manifest.json`. Загружаются пять файлов v3, контрольные вопросы, конфигурация и ответы baseline, конфигурация судьи и три отчёта аудита. Совпадающие файлы не скачиваются повторно. При конфликте общих контрольных файлов выполнение останавливается; изменяемые файлы v3 предварительно сохраняются в резервную копию.

```bash
python scripts/audit_sft_v3.py
python scripts/sync_colab_data.py --project-root . --verify-local
python -m unittest discover -s tests -v
```

Для намеренной пересборки v3:

```bash
python scripts/build_sft_v3.py
python scripts/update_v3_manifest.py
```

Сборщик пишет только в каталоги v3. Более ранние данные сохраняются. Скрипт `scripts/export_notebook_evidence.py` извлекает таблицы, приведённые в PDF, из зафиксированной в его коде версии ноутбука в Git; он требует полной истории репозитория. Результаты такого экспорта уже опубликованы в каталогах без `_recovery`.

Расхождения PDF с доступными материалами перечислены в [сверке артефактов](docs/REPORT_ARTIFACT_CHECK.md). PDF сохранён без изменений.
