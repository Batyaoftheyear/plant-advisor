"""Assemble report-only reproducibility materials for revised synthetic SFT data."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'results'/'dataset'
rows=[json.loads(s) for s in (DATA/'sft_metadata.jsonl').read_text(encoding='utf-8').splitlines() if s]
stats=json.loads((OUT/'dataset_audit.json').read_text(encoding='utf-8'))
original_stats=json.loads((OUT/'revision_summary.json').read_text(encoding='utf-8'))

# Only links referenced by at least one actual record. All are independently
# identifiable public articles/guides; sentence-by-sentence attribution is not implied.
sources=pd.read_csv(DATA/'sources.csv')
used=set(stats['used_source_ids'])
with (OUT/'source_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f)
 w.writerow(['source_id','title','url','used_by_records','scope','review_limit'])
 for _, src in sources[sources['source_id'].isin(used)].iterrows():
  w.writerow([src['source_id'],src['source_name'],src['url'],sum(src['source_id'] in r['source_ids'] for r in rows),
              'general horticultural guidance',
              'Article identity and general topic checked; not a line-by-line evidence audit of every example'])

counts=(pd.DataFrame(rows).groupby(['split','task']).size().unstack(fill_value=0).reindex(['train','validation']))
counts.to_csv(OUT/'split_distribution.csv')
fig, axes=plt.subplots(1,2,figsize=(11,4.3))
counts.plot(kind='bar',rot=0,ax=axes[0])
axes[0].set_title('Примеры по категориям')
axes[0].set_xlabel('Выборка')
axes[0].set_ylabel('Количество')
word_counts=[len(r['answer'].split()) for r in rows]
axes[1].hist(word_counts,bins=16)
axes[1].set_title('Длина ответов')
axes[1].set_xlabel('Количество слов')
axes[1].set_ylabel('Частота')
fig.tight_layout()
fig.savefig(OUT/'dataset_distributions.png',dpi=150,bbox_inches='tight')
plt.close(fig)

# Attach an honest revised-split summary in the stage report.
report=f'''# PlantAdvisor — повторная ревизия SFT-датасета

## Проверенный комплект

В работе использованы **400 синтетических диалогов** из первой версии PlantAdvisor. Участки с предметными и языковыми ошибками исправлены с сохранением ID, назначения выборки и формата `messages`.

| Проверка | До ревизии | После ревизии |
|---|---:|---:|
| Всего записей | 400 | {stats['record_count']} |
| Train / validation | 360 / 40 | {stats['train_count']} / {stats['validation_count']} |
| Care / diagnosis | 200 / 200 | 200 / 200 |
| Уникальные вопросы | 400 | {stats['unique_questions']} |
| Уникальные ответы | 289 | {stats['unique_answers']} |
| Повторы одинаковых ответов | {stats['original_duplicate_answers']} | {stats['revised_duplicate_answers']} |
| Точные совпадения с тестовыми вопросами | 0 | {stats['exact_eval_question_duplicates']} |

**Исправлено {stats['revised_count']} записей.** Изменения отмечены в `results/dataset/revision_log.csv` (исходный и новый вопрос/ответ, причина). Категории правок могут пересекаться: предметные исправления — {stats['change_categories'].get('botanical',0)}, грамматические — {stats['change_categories'].get('grammar',0)}, улучшение формулировок — {stats['change_categories'].get('wording',0)}, видоспецифичность — {stats['change_categories'].get('specificity',0)}.

### Основные исправления

- Вопросы о **пустынном кактусе** переведены с некорректных «листьев» на стебель, приросты, ареолы и корни, а ответы пересмотрены применительно к этим органам (`SFT-0121`–`SFT-0128`).
- Исправлены ошибочные падежи в вопросах «на хлорофитуму», «на фаленопсису», «на сансевиерией» и т.п.; вместо неестественного «в содержании монстеры» используются нейтральные формулировки.
- В полностью одинаковых ответах по подкормкам, пятнам, вредителям, увяданию и мягкому основанию добавлены уместные особенности ухода за конкретным видом. **Это устраняет точные повторы, но не все смысловые/шаблонные повторения.**
- Для фаленопсиса уточнена терминология: кора/субстрат вместо обычной почвы, где это существенно.

## Утечки и сходство

Сопоставлены все 400 вопросов с отдельными 30 вопросами контрольного набора. Точных совпадений нет. Максимальное символьное TF–IDF-сходство: **{stats['eval_tf_idf_max']:.3f}**, совпадений с порогом ≥0,85: **{stats['eval_tf_idf_ge_0.85']}**. Между train и validation максимальное сходство вопросов **{stats['cross_split_question_tf_idf_max']:.3f}**, пар с порогом ≥0,85: **{stats['cross_split_pairs_ge_0.85']}**. Это не исключает смысловую близость типовых ситуаций.

## Источники и ограничения проверки

В датасете используются {len(stats['used_source_ids'])} уникальных URL из RHS, University of Maryland Extension и University of Minnesota Extension. Они относятся к соответствующей **общей тематике**; на основании одного факта наличия URL нельзя утверждать, что каждый совет в каждом ответе отдельно подтверждён публикацией. Список использованных ссылок и назначение приведены в `results/dataset/source_review.csv`.

Все 400 записей прошли **автоматические структурные и межфайловые проверки**, а шаблоны и рискованные сценарии прошли адресную содержательную ревизию. Полная независимая экспертная оценка 400 ботанических консультаций **не проводилась**. В `results/dataset/manual_review_sample.csv` предусмотрена независимая ручная проверка 30 примеров; соответствующие колонки оставлены пустыми.

Обучение на этом датасете — учебный эксперимент. Возможный прирост по формату не означает гарантированного улучшения диагностической точности. Желательна независимая проверка части ответов, особенно диагностических.

## Совместимость и воспроизведение

Сохранены старые пути и структура файлов:

- `data/train.jsonl` — 360 записей;
- `data/validation.jsonl` — 40 записей;
- `data/sft_metadata.jsonl`, `data/sft_provenance.csv`, `data/sources.csv` — происхождение данных;
- `data/eval_questions.jsonl` — неизменённый тестовый набор.

Для повторения ревизии из сохранённого исходного набора:

```bash
python scripts/revise_sft_dataset.py
python scripts/final_audit_sft.py
```

Оба скрипта работают с `data/sft_metadata_raw.jsonl`; повторный запуск не вносит дополнительные изменения. `02_dataset.ipynb` читает прежние пути без изменения кода.
'''
(ROOT/'REVISION_REPORT.md').write_text(report,encoding='utf-8')

# Data-card update deliberately distinguishes original synthetic generation and second pass.
card=(ROOT/'DATASET_CARD.md').read_text(encoding='utf-8')
card=card.replace('## Как сформированы записи', '''## Повторная ревизия (v2)

После первоначального формирования проведён полный автоматический аудит 400 строк, адресная предметная и грамматическая правка 174 записей, удалены 111 точных повторов ответов путём добавления относящихся к конкретному виду проверок или корректировки ошибок. Изменения воспроизводятся скриптом `scripts/revise_sft_dataset.py`; подробности — в `REVISION_REPORT.md`. Отдельный файл `data/sft_metadata_raw.jsonl` хранит первоначальную версию для воспроизводимости. Независимая экспертная проверка всех 400 записей не проводилась.

## Как сформированы записи''')
card=card.replace('Для воспроизведения генерации при наличии `data/eval_questions.jsonl` из корня проекта:', 'Для воспроизведения первоначальной генерации при наличии `data/eval_questions.jsonl` из корня проекта:')
card=card.replace('''python scripts/build_sft_dataset.py
```''','''python scripts/build_sft_dataset.py
python scripts/revise_sft_dataset.py
python scripts/final_audit_sft.py
```''')
(ROOT/'DATASET_CARD.md').write_text(card,encoding='utf-8')
readme=(ROOT/'README_DATASET.md').read_text(encoding='utf-8')
readme='> Обновление v2: 174 исправленные записи, нет полностью одинаковых ответов; подробности в `REVISION_REPORT.md`. Формат файлов не изменился.\n\n'+readme
(ROOT/'README_DATASET.md').write_text(readme,encoding='utf-8')

print(report[:1000]);print('support files ready')
