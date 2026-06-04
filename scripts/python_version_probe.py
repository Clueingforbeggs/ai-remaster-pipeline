from __future__ import annotations

import sys
import sysconfig

print(f"{sys.version_info.major}.{sys.version_info.minor}")
print(f"implementation={sys.implementation.name}")
print(f"bits={sys.maxsize.bit_length() + 1}")
print(f"gil_disabled={1 if sysconfig.get_config_var('Py_GIL_DISABLED') else 0}")
