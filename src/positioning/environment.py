"""Environment helpers for acoustic propagation."""


def speed_of_sound(temperature_c: float = 20.0, humidity_percent: float = 0.0) -> float:
    """Return an approximate speed of sound in air, m/s.

    The humidity term is intentionally simple: it is enough for the current
    prototype and keeps the model transparent for the diploma documentation.
    """
    if not -80.0 <= temperature_c <= 80.0:
        raise ValueError("temperature_c must be in the range [-80, 80]")
    if not 0.0 <= humidity_percent <= 100.0:
        raise ValueError("humidity_percent must be in the range [0, 100]")
    return 331.3 + 0.606 * temperature_c + 0.0124 * humidity_percent
