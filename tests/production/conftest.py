"""The production code lives outside the package, in ``scripts/production`` (README.md there): its directory goes on
``sys.path`` for these tests, as ``run_production.py`` puts it there for itself."""

import sys

from skewerjax.utils.paths import PRODUCTION_SCRIPTS_DIR

if str(PRODUCTION_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_SCRIPTS_DIR))
