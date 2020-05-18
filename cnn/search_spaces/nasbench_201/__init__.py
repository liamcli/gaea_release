import sys
from pathlib import Path

lib_dir = "/code/AutoDL/lib"
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
