"""Early-loaded pytest plugin bridging HA core dev and the released PHCC plugin.

pytest-homeassistant-custom-component is generated from the latest HA release,
so it can lag behind HA core@dev (the CI "Test (HA dev)" job). Load with
`-p tests.ha_dev_compat` so this runs before PHCC's plugin is imported.
"""

from homeassistant.components import logger

# HA core 2026.11 (home-assistant/core#187220) moved _clear_logger_overwrites
# to logger.helpers and stopped re-exporting it, but PHCC's common.py still
# imports it from the package. Restore the old name until PHCC is regenerated
# from a 2026.11 release.
if not hasattr(logger, "_clear_logger_overwrites"):
    from homeassistant.components.logger.helpers import _clear_logger_overwrites

    logger._clear_logger_overwrites = _clear_logger_overwrites  # type: ignore[attr-defined]
