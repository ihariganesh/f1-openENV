"""Race Strategy Optimizer package (alternate implementation path).

Note:
The deployed FastAPI app for this submission imports root modules
(`env.py`, `models.py`, `tasks.py`). This package is retained for
package-style development and tests, but is not the primary runtime entrypoint.
"""

from .environment import RaceStrategyEnvironment

__all__ = ["RaceStrategyEnvironment"]
