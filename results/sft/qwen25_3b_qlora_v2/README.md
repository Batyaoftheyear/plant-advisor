# QLoRA: первый эксперимент

Базовая модель — `Qwen/Qwen2.5-3B-Instruct`, ревизия `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Данные: `data/train.jsonl` и `data/validation.jsonl`, 360/40 записей. SHA-256 входов указан в `training_config.json`.

Параметры: 3 эпохи, LR `1e-4`, LoRA `r=16`, `alpha=32`, dropout `0.05`, NF4/FP16. Лучший checkpoint — шаг 45 с `eval_loss=1.8348503112792969`. На шагах 90 и 135 validation loss вырос до 2.0100388526916504 и 2.021094560623169.

`trainer_state.json` содержит журнал до шага 135, `training_state_checkpoint45.json` — состояние лучшего checkpoint. `training_log.csv` представляет записи `log_history` в табличном виде. Происхождение и SHA-256 файлов указаны в `drive_artifacts_manifest.json`.

[Веса и токенизатор checkpoint-45](https://github.com/Batyaoftheyear/plant-advisor/releases/tag/stage1-v2-checkpoint45). Адаптер содержит 504 тензора F32 и 29 933 568 параметров.
