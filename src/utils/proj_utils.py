from __future__ import annotations

import os
from pathlib import Path


def configure_proj_data_dir() -> str:
    candidates: list[str] = []
    for key in ("PROJ_DATA", "PROJ_LIB"):
        value = os.environ.get(key)
        if value:
            candidates.append(value)

    pyproj = None
    try:
        import pyproj  # type: ignore

        pkg_dir = Path(pyproj.__file__).resolve().parent
        candidates.extend(
            [
                str(pkg_dir / "proj_dir" / "share" / "proj"),
                str(pkg_dir / "data"),
                str(pkg_dir / "proj_data"),
            ]
        )
    except Exception:
        pyproj = None

    candidates.extend(
        [
            "/usr/share/proj",
            "/usr/local/share/proj",
            "/opt/conda/share/proj",
            "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj",
        ]
    )

    chosen = next((path for path in candidates if path and (Path(path) / "proj.db").exists()), None)
    if not chosen:
        raise RuntimeError(
            "PROJ data directory not found. Set PROJ_DATA or PROJ_LIB to a folder containing proj.db."
        )

    os.environ["PROJ_DATA"] = chosen
    os.environ["PROJ_LIB"] = chosen

    if pyproj is not None:
        from pyproj import datadir

        datadir.set_data_dir(chosen)

    return chosen
