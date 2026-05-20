# Positioning

Python-прототип пассивной акустической локализации источника звука по TDOA для компланарной решетки из четырех направленных микрофонов.

Проект используется как практическая часть дипломной работы: он формирует воспроизводимый программный стенд для синтетических сигналов, предобработки, проверки задержек, локализации направления/координат, сопровождения цели и протоколирования результатов ПМИ.

## Быстрый запуск

Команды выполняются из папки `positioning`.

```powershell
python -m pip install -e ".[dev,docs]"

python .\tools\demo.py
python .\tools\validate.py
python -m pytest
python .\tools\export_docs_to_pdf.py
streamlit run .\src\positioning\streamlit_app.py
```

Расширенный сценарий с файлами, журналами и ПМИ:

```powershell
positioning inspect-audio --input .\data\synthetic\side_looking_point_pulse.wav --require-metadata
positioning preprocess --input .\data\synthetic\side_looking_point_pulse.csv --output-dir .\data\processed
positioning preprocess --input .\data\synthetic\side_looking_good_trajectory.csv --output-dir .\data\processed_demo
positioning localization --input .\data\synthetic\direction_near_curve.wav --mode position --show-visualization --trajectory-max-windows 18
positioning localization --input .\data\synthetic\direction_far_curve.wav --mode direction --show-visualization --trajectory-max-windows 24
positioning localize-audio --input .\data\synthetic\side_looking_point_pulse.wav --mode direction --max-windows 4 --csv-log .\data\logs\audio_localization.csv --json-output .\data\logs\audio_final_result.json
positioning demo --json-output .\data\logs\demo_result.json
positioning track-demo --csv-log .\data\logs\tracking.csv --json-output .\data\logs\final_result.json --inject-low-confidence-jump
positioning pmi --output-dir .\data\pmi --repeat-count 100
streamlit run .\src\positioning\streamlit_app.py
```

## Основная структура

- `src/positioning/` - активный Python-пакет.
- `configs/default.json` - конфигурация решетки, аудио, среды и режима локализации.
- `configs/side_looking_demo.json` - рабочая конфигурация боковой компланарной решетки для демо/Streamlit.
- `tools/demo.py` - удобная обертка для демонстрации.
- `tools/validate.py` - удобная обертка для синтетического валидатора.
- `tests/` - pytest-набор.
- `docs/` - подробная документация проекта.
- `docs/pdf/` - PDF-экспорт документации.
- `data/` - генерируемые сигналы, журналы и протоколы, не отслеживаются Git.

## Что реализовано

- Формальные модели данных: микрофонная решетка, среда, аудиофрейм, TDOA-оценка, результат локализации.
- JSON-конфигурация без внешних зависимостей.
- Компланарная квадратная решетка в боковой конфигурации (плоскость `yz`, нормаль вдоль `x`).
- Генерация синтетических 4-канальных сигналов с шумом, ослаблением, выбросом и отраженным путем.
- Генерация синтетической траектории движущегося источника с массивом `trajectory` в metadata.
- Поддержка неимпульсных источников: `chirp`, `multitone_burst`, `band_noise`, `am_fm_tone`.
- Чтение и запись 4-канальных CSV/WAV-файлов и sidecar-метаданных.
- Разбиение на окна, удаление DC, нормализация, Hann-окно, частотная маска и диагностика информативности.
- Проверка TDOA: физическая допустимость, транзитивность, качество пиков, робастный перебор подмножеств.
- Оценка TDOA из аудиоокон через GCC-PHAT, включая субполосный режим (`subbands`) и temporal delay continuity (`unwrap`).
- Локализация в режимах `direction` и `position`.
- Сглаживание во времени: EMA азимута/угла места, отбраковка скачков, угловая скорость, агрегированная уверенность.
- JSON-вывод результата и CSV-журнал по окнам.
- Streamlit-интерфейс с интерактивной 3D-визуализацией режимов `direction` и `position` (в `direction` сравнение идет по углам на фиксированном радиусе).
- Автоматизированный прогон ПМИ с Markdown/JSON/CSV-протоколами.

## Важное ограничение

GCC-PHAT уже подключен к файловому конвейеру `CSV/WAV -> окна -> предобработка -> TDOA -> локализация`. Для реальных записей остаются практические вопросы калибровки: синхронность каналов, известная ориентация направленных микрофонов, отношение сигнал/шум, отражения помещения и подбор частотной полосы.

Подробности:

- `docs/PROJECT_OVERVIEW.md` - архитектура и конвейер.
- `docs/USER_GUIDE.md` - команды запуска и форматы файлов.
- `docs/ALGORITHM_TDOA_DETAILED.md` - математическая логика TDOA, робастность и режимы.
- `docs/MODEL_STATUS.md` - готовность модулей и ограничения.
- `docs/PMI_PROTOCOL_SUMMARY.md` - таблица испытаний по ПМИ.
- `docs/CODEBASE_AUDIT.md` - ревизия алгоритмов и кандидатов на удаление.
