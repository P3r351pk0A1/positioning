# Документация проекта Positioning

Проект `positioning` - Python-прототип пассивной акустической локализации по TDOA для компланарной решетки из 4 микрофонов.

## Назначение

Рабочий конвейер практической части:

```text
конфигурация -> сигнал (synthetic/CSV/WAV) -> окна -> предобработка -> GCC-PHAT -> проверки TDOA -> локализация -> трекинг -> JSON/CSV/ПМИ
```

## Структура каталогов

```text
positioning/
  configs/
    default.json
    side_looking_demo.json
  docs/
  src/
    positioning/
      cli.py
      config.py
      environment.py
      gcc_phat.py
      geometry.py
      localization.py
      models.py
      offline_io.py
      output.py
      pmi.py
      preprocess.py
      simulation.py
      streamlit_app.py
      tdoa.py
      tracking.py
      validation.py
      visualization.py
  tests/
  tools/
```

## Основные алгоритмические блоки

- `preprocess.py`: DC removal, Hann window, частотная маска, диагностика информативности окна.
- `gcc_phat.py`: оценка межканальных задержек:
  - PHAT-нормировка,
  - интерполяция пика,
  - режим `subbands` (консенсус по полосам),
  - `delay continuity` (unwrap по времени),
  - fallback на full-band для узкополосного сигнала.
- `tdoa.py`: физическая допустимость и транзитивность задержек.
- `localization.py`: режимы `direction` и `position`, робастная отбраковка пар, остаточная ошибка, confidence/reliability.
- `tracking.py`: сглаживание результатов по окнам (EMA), подавление резких низкоуверенных скачков.
- `output.py`: JSON и CSV журнал.
- `pmi.py`: автоматизированный протокол ПМИ.

## Синтетические сигналы

`simulation.py` поддерживает:

- `sine`
- `pulse`
- `chirp`
- `multitone_burst`
- `band_noise`
- `am_fm_tone`

Для траекторий в metadata пишется массив `trajectory` с timestamp и координатами.

## Визуализация

- `streamlit_app.py` - основной интерфейс.
- В режиме `direction` отображение идет на фиксированном радиусе (`Direction ray length`), то есть сравниваются углы направления, а не псевдодальность.
- В таблице окон есть `truth_angle_error_deg` для угловой проверки качества в `direction`.

## Практический статус

- Пайплайн GCC-PHAT для офлайн CSV/WAV реализован.
- Тесты: `python -m pytest` (текущий набор проходит).
- ПМИ автоматизирован (`positioning pmi`) и формирует протоколы в `data/pmi/`.
