"""Put `scripts/` on the import path.

`scripts/` is not a package -- the scripts are entry points, run as
`uv run python scripts/probe_mrl.py`. But `probe_mrl.py` carries the probe's
statistic and its reductions, and those are exactly the things worth testing
without a model attached, so the test suite has to be able to import it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
