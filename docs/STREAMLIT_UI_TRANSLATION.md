# Streamlit UI translation log

Format: English - Russian - Location

| English | Russian | Location |
| --- | --- | --- |
| TDOA Positioning | Позиционирование по TDOA | src/positioning/streamlit_app.py (main: page_title) |
| TDOA Positioning Dashboard | Панель позиционирования по TDOA | src/positioning/streamlit_app.py (main: title) |
| Input | Входные данные | src/positioning/streamlit_app.py (_sidebar) |
| Config JSON | Конфигурация JSON | src/positioning/streamlit_app.py (_sidebar) |
| Signal CSV/WAV | Сигнал CSV/WAV | src/positioning/streamlit_app.py (_sidebar) |
| Metadata JSON | Метаданные JSON | src/positioning/streamlit_app.py (_sidebar) |
| Mode | Режим | src/positioning/streamlit_app.py (_sidebar) |
| Localization mode | Режим локализации | src/positioning/streamlit_app.py (_sidebar) |
| direction | направление | src/positioning/streamlit_app.py (_mode_label) |
| position | позиция | src/positioning/streamlit_app.py (_mode_label) |
| Windows | Окна | src/positioning/streamlit_app.py (_sidebar) |
| trajectory | траектория | src/positioning/streamlit_app.py (_window_selection_label) |
| first | первые | src/positioning/streamlit_app.py (_window_selection_label) |
| Max windows | Макс. число окон | src/positioning/streamlit_app.py (_sidebar) |
| GCC-PHAT interpolation | Интерполяция GCC-PHAT | src/positioning/streamlit_app.py (_sidebar) |
| GCC subbands | Субполосы GCC | src/positioning/streamlit_app.py (_sidebar) |
| TDOA policy | Политика TDOA | src/positioning/streamlit_app.py (_sidebar) |
| auto | авто | src/positioning/streamlit_app.py (_tdoa_policy_mode_label) |
| manual | вручную | src/positioning/streamlit_app.py (_tdoa_policy_mode_label) |
| Min pair quality | Мин. качество пары | src/positioning/streamlit_app.py (_sidebar) |
| Min result confidence | Мин. уверенность результата | src/positioning/streamlit_app.py (_sidebar) |
| Show low_confidence | Показывать низкую уверенность | src/positioning/streamlit_app.py (_sidebar) |
| Delay continuity (unwrap) | Непрерывность задержки (развертка) | src/positioning/streamlit_app.py (_sidebar) |
| Display max speed, m/s | Макс. скорость отображения, м/с | src/positioning/streamlit_app.py (_sidebar) |
| 3D Display | 3D-отображение | src/positioning/streamlit_app.py (_sidebar) |
| Direction ray length, m | Длина луча направления, м | src/positioning/streamlit_app.py (_sidebar) |
| Truth trajectory | Истинная траектория | src/positioning/streamlit_app.py (_sidebar, _build_figure) |
| Position estimates | Оценки положения | src/positioning/streamlit_app.py (_sidebar) |
| Direction rays | Лучи направления | src/positioning/streamlit_app.py (_sidebar) |
| Custom path... | Свой путь... | src/positioning/streamlit_app.py (_path_selectbox) |
| {label} path | {label}: путь | src/positioning/streamlit_app.py (_path_selectbox) |
| window is not informative: ... | окно неинформативно: ... | src/positioning/streamlit_app.py (_estimate_windows) |
| Channels | Каналы | src/positioning/streamlit_app.py (_summary) |
| Selected | Выбрано | src/positioning/streamlit_app.py (_summary) |
| OK | Успешно | src/positioning/streamlit_app.py (_summary) |
| Shown | Показано | src/positioning/streamlit_app.py (_summary) |
| TDOA policy mix: narrowband=..., wideband=... | Баланс политики TDOA: узкополосный=..., широкополосный=... | src/positioning/streamlit_app.py (_summary) |
| RMS error (N points): ... | СКО ошибки (N точек): ... | src/positioning/streamlit_app.py (_summary) |
| Array center: ... | sound speed: ... m/s | truth points: ... | informative selected windows: ... | Центр решетки: ... | скорость звука: ... м/с | опорные точки: ... | информативных окон: ... | src/positioning/streamlit_app.py (_summary) |
| Trajectory truncated by window limit: ... Increase Max windows to show full trajectory. | Траектория усечена лимитом окон: ... Увеличьте лимит окон, чтобы показать всю траекторию. | src/positioning/streamlit_app.py (_summary) |
| High zero ratio in source signal: ... For pulse-based synthetic scenarios this is expected (short impulses in long silence). | Высокая доля нулей в исходном сигнале: ... Для синтетики с импульсами это ожидаемо (короткие импульсы и длинные паузы). | src/positioning/streamlit_app.py (_summary) |
| Current Result | Текущий результат | src/positioning/streamlit_app.py (_result_panel) |
| No accepted estimates. | Нет принятых оценок. | src/positioning/streamlit_app.py (_result_panel) |
| Status | Статус | src/positioning/streamlit_app.py (_result_panel) |
| Confidence | Уверенность | src/positioning/streamlit_app.py (_result_panel) |
| Timestamp | Время | src/positioning/streamlit_app.py (_result_panel) |
| Azimuth | Азимут | src/positioning/streamlit_app.py (_result_panel) |
| Elevation | Угол места | src/positioning/streamlit_app.py (_result_panel) |
| Position: | Положение: | src/positioning/streamlit_app.py (_result_panel) |
| Direction: | Направление: | src/positioning/streamlit_app.py (_result_panel) |
| Window Estimates | Оценки по окнам | src/positioning/streamlit_app.py (_table) |
| window | окно | src/positioning/streamlit_app.py (_table) |
| timestamp | время, с | src/positioning/streamlit_app.py (_table) |
| status | статус | src/positioning/streamlit_app.py (_table) |
| confidence | уверенность | src/positioning/streamlit_app.py (_table) |
| azimuth_deg | азимут, град | src/positioning/streamlit_app.py (_table) |
| elevation_deg | угол места, град | src/positioning/streamlit_app.py (_table) |
| x | x, м | src/positioning/streamlit_app.py (_table) |
| y | y, м | src/positioning/streamlit_app.py (_table) |
| z | z, м | src/positioning/streamlit_app.py (_table) |
| truth_error_m | ошибка по истине, м | src/positioning/streamlit_app.py (_table) |
| truth_angle_error_deg | угловая ошибка, град | src/positioning/streamlit_app.py (_table) |
| used_pairs | использованные пары | src/positioning/streamlit_app.py (_table) |
| rejected_pairs | отброшенные пары | src/positioning/streamlit_app.py (_table) |
| message | сообщение | src/positioning/streamlit_app.py (_table) |
| tdoa_policy | политика TDOA | src/positioning/streamlit_app.py (_table) |
| Microphones | Микрофоны | src/positioning/streamlit_app.py (_build_figure) |
| Array plane | Плоскость решетки | src/positioning/streamlit_app.py (_build_figure) |
| Displayed estimates | Показанные оценки | src/positioning/streamlit_app.py (_build_figure) |
| Direction ray | Луч направления | src/positioning/streamlit_app.py (_build_figure) |
| X, m | X, м | src/positioning/streamlit_app.py (_build_figure) |
| Y, m | Y, м | src/positioning/streamlit_app.py (_build_figure) |
| Z, m | Z, м | src/positioning/streamlit_app.py (_build_figure) |
| t=...s | t=... с | src/positioning/streamlit_app.py (_hover_text) |
| status= | статус= | src/positioning/streamlit_app.py (_hover_text) |
| confidence= | уверенность= | src/positioning/streamlit_app.py (_hover_text) |
| azimuth=... deg | азимут=... град | src/positioning/streamlit_app.py (_hover_text) |
| elevation=... deg | угол места=... град | src/positioning/streamlit_app.py (_hover_text) |
| n/a | н/д | src/positioning/streamlit_app.py (_fmt_optional) |
| deg | град | src/positioning/streamlit_app.py (_rms_effectiveness_metric, _result_panel, _hover_text) |
| m | м | src/positioning/streamlit_app.py (_rms_effectiveness_metric, _summary, _build_figure) |
| m/s | м/с | src/positioning/streamlit_app.py (_summary, _sidebar) |
| s | с | src/positioning/streamlit_app.py (_result_panel, _hover_text) |
| ok | ок | src/positioning/streamlit_app.py (_status_label) |
| low_confidence | низкая уверенность | src/positioning/streamlit_app.py (_status_label) |
| insufficient_signal | недостаточный сигнал | src/positioning/streamlit_app.py (_status_label) |
| invalid_input | некорректные данные | src/positioning/streamlit_app.py (_status_label) |
| narrowband | узкополосный | src/positioning/streamlit_app.py (_tdoa_policy_label, _summary) |
| wideband | широкополосный | src/positioning/streamlit_app.py (_tdoa_policy_label, _summary) |
| manual | ручной | src/positioning/streamlit_app.py (_tdoa_policy_label) |
