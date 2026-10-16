"""CUDA add-on wheel: only the CUDA plugin, installed into OpenMM.libs/lib/plugins."""

import os
import sys
import tempfile
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from versions import load_versions, release_version, requirement, source_version  # noqa: E402

minor = os.environ["OC_OPENMM_MINOR"]
cuda = os.environ["OC_CUDA"]
entry = load_versions()[minor]
version = release_version(source_version(os.environ["OC_SOURCE_DIR"]), entry["post"])


class PlatformWheel(bdist_wheel):
    # The plugin does not depend on the Python version, only on the platform
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", os.environ["OC_PLATFORM"]


setup(
    name=f"openmm-colvars-cuda-{cuda}",
    version=version,
    description=f"CUDA {cuda} kernels for openmm-colvars",
    author="OpenMM-Colvars contributors",
    url="https://github.com/fhh2626/OpenMM-Colvars",
    license="GPL-3.0-or-later",
    python_requires=">=3.10",
    py_modules=[],
    install_requires=[requirement(minor, entry, f"openmm-cuda-{cuda}"), f"openmm-colvars=={version}"],
    cmdclass={"bdist_wheel": PlatformWheel},
    # A fresh build directory: setuptools would reuse files built from an earlier staging tree
    options={"build": {"build_base": tempfile.mkdtemp(prefix="openmm-colvars-build-")}},
)
