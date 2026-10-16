"""Base wheel: Python modules, extension and libOpenMMColvars, without the CUDA plugin."""

import os
import sys
import tempfile
from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from versions import load_versions, release_version, requirement, source_version  # noqa: E402

minor = os.environ["OC_OPENMM_MINOR"]
entry = load_versions()[minor]
version = release_version(source_version(os.environ["OC_SOURCE_DIR"]), entry["post"])
readme = Path(os.environ["OC_SOURCE_DIR"]) / "README.md"


class BinaryDistribution(Distribution):
    # The extension and library are injected after the build; this makes the wheel platform-specific
    def has_ext_modules(self):
        return True


setup(
    name="openmm-colvars",
    version=version,
    description="Colvars collective variables and NAMD-like FEP for OpenMM",
    long_description=readme.read_text() if readme.exists() else "",
    long_description_content_type="text/markdown",
    author="OpenMM-Colvars contributors",
    url="https://github.com/fhh2626/OpenMM-Colvars",
    license="GPL-3.0-or-later",
    python_requires=">=3.10",
    package_dir={"": os.environ["OC_STAGE_DIR"]},
    py_modules=["openmm_colvars"],
    packages=["openmm_fep"],
    install_requires=[requirement(minor, entry)],
    extras_require={
        f"cuda{cuda}": [f'openmm-colvars-cuda-{cuda}=={version}; platform_system == "Linux"']
        for cuda in (12, 13)
    },
    distclass=BinaryDistribution,
    # A fresh build directory: setuptools would reuse files built from an earlier staging tree
    options={"build": {"build_base": tempfile.mkdtemp(prefix="openmm-colvars-build-")}},
)
