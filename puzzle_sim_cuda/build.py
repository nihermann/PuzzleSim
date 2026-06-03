#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convenience entry point for building the CUDA extension.

This is a thin wrapper around ``python setup.py build_ext --inplace`` that
keeps the working directory pinned to the location of ``setup.py``.
"""

import argparse
import os
import shutil
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true",
                        help="Remove build artifacts before building")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose build output")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    if args.clean:
        for d in ("build", "dist"):
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"removed {d}/")

    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    if args.verbose:
        cmd.append("--verbose")
    rc = _run(cmd)
    if rc != 0:
        print("Build failed.", file=sys.stderr)
        return rc

    print("\nBuild complete. Quick sanity check:")
    rc = _run([sys.executable, "-c",
               "import torch, puzzle_sim_cuda_ext as e; "
               "print('loaded:', sorted([s for s in dir(e) if not s.startswith('_')]))"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
