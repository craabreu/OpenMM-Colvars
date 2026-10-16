import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import package  # noqa: E402

NVIDIA = "$ORIGIN/../../../nvidia"


def test_linux_runtime_paths():
    assert package.runtime_paths("_openmm_colvars.cpython-312-x86_64-linux-gnu.so", "Linux", None) == [
        "$ORIGIN/OpenMM.libs/lib"]
    assert package.runtime_paths("OpenMM.libs/lib/libOpenMMColvars.so", "Linux", None) == ["$ORIGIN"]
    assert package.runtime_paths("OpenMM.libs/lib/plugins/libOpenMMColvarsCUDA.so", "Linux", None) == [
        "$ORIGIN", "$ORIGIN/.."]


def test_linux_cuda_runtime_paths():
    plugin = "OpenMM.libs/lib/plugins/libOpenMMColvarsCUDA.so"
    assert package.runtime_paths(plugin, "Linux", 12) == [
        "$ORIGIN", "$ORIGIN/..", f"{NVIDIA}/cufft/lib", f"{NVIDIA}/cuda_nvrtc/lib", f"{NVIDIA}/cuda_runtime/lib"]
    assert package.runtime_paths(plugin, "Linux", 13) == ["$ORIGIN", "$ORIGIN/..", f"{NVIDIA}/cu13/lib"]


def test_macos_runtime_paths():
    assert package.runtime_paths("_openmm_colvars.cpython-312-darwin.so", "Darwin", None) == [
        "@loader_path/OpenMM.libs/lib"]
    assert package.runtime_paths("OpenMM.libs/lib/libOpenMMColvars.dylib", "Darwin", None) == ["@loader_path"]


def _wheel(path, names):
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"")
    return path


BASE = [
    "openmm_colvars.py",
    "_openmm_colvars.cpython-312-x86_64-linux-gnu.so",
    "openmm_fep/__init__.py",
    "OpenMM.libs/lib/libOpenMMColvars.so",
]
REQUIRED = ["openmm_colvars.py", "_openmm_colvars*", "openmm_fep/__init__.py", "OpenMM.libs/lib/libOpenMMColvars.*"]


def test_check_wheel_accepts_our_files(tmp_path):
    package.check_wheel(_wheel(tmp_path / "ok.whl", BASE), REQUIRED)


def test_check_wheel_rejects_grafted_libraries(tmp_path):
    wheel = _wheel(tmp_path / "bad.whl", BASE + ["openmm_colvars.libs/libOpenMM-1a2b.so"])
    with pytest.raises(SystemExit, match="bundled"):
        package.check_wheel(wheel, REQUIRED)


def test_check_wheel_rejects_foreign_openmm_files(tmp_path):
    wheel = _wheel(tmp_path / "bad.whl", BASE + ["OpenMM.libs/lib/libOpenMM.so"])
    with pytest.raises(SystemExit, match="foreign"):
        package.check_wheel(wheel, REQUIRED)


def test_check_wheel_requires_the_core_library(tmp_path):
    wheel = _wheel(tmp_path / "nocore.whl", [n for n in BASE if "libOpenMMColvars" not in n])
    with pytest.raises(SystemExit, match="missing"):
        package.check_wheel(wheel, REQUIRED)


LDD = (
    "\tlibOpenMM.so => /x/OpenMM.libs/lib/libOpenMM.so (0x1)\n"
    "\tlibcuda.so.1 => not found\n"
    "\tlibcufft.so.11 => /x/nvidia/cufft/lib/libcufft.so.11 (0x2)\n"
    "undefined symbol: cuInit\t(/x/OpenMM.libs/lib/plugins/libOpenMMColvarsCUDA.so)\n"
    "undefined symbol: cuMemAlloc_v2\t(/x/OpenMM.libs/lib/plugins/libOpenMMCUDA.so)\n"
)


def test_link_problems_allow_only_the_driver():
    # Without a GPU, only libcuda.so.1 and the driver API (cuXxx) may be unresolved
    assert package.link_problems(LDD) == []


def test_link_problems_report_runtime_mismatches():
    output = LDD + (
        "\tlibnvrtc.so.12 => not found\n"
        "undefined symbol: cudaGraphInstantiateWithParams\t(/x/libOpenMMColvarsCUDA.so)\n"
        "undefined symbol: _ZN6OpenMM10CudaContext5fooEv\t(/x/libOpenMMColvarsCUDA.so)\n"
        "/x/libOpenMMColvarsCUDA.so: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.32' not found\n"
    )
    assert package.link_problems(output) == [
        "missing library libnvrtc.so.12",
        "undefined symbol cudaGraphInstantiateWithParams",
        "undefined symbol _ZN6OpenMM10CudaContext5fooEv",
        "/x/libOpenMMColvarsCUDA.so: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.32' not found",
    ]


def test_inject_adds_staged_files(tmp_path):
    stage = tmp_path / "stage"
    (stage / "OpenMM.libs/lib").mkdir(parents=True)
    (stage / "openmm_fep/__pycache__").mkdir(parents=True)
    (stage / "openmm_colvars.py").write_text("staged")
    (stage / "openmm_fep/__init__.py").write_text("staged")
    (stage / "openmm_fep/__pycache__/__init__.cpython-312.pyc").write_bytes(b"")
    (stage / "_openmm_colvars.cpython-312-x86_64-linux-gnu.so").write_bytes(b"ext")
    (stage / "OpenMM.libs/lib/libOpenMMColvars.so").write_bytes(b"lib")
    wheel = tmp_path / "x-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("openmm_colvars.py", "packaged")
        archive.writestr("openmm_fep/__init__.py", "packaged")
        archive.writestr("x-1.0.dist-info/RECORD",
                         "openmm_colvars.py,,\nopenmm_fep/__init__.py,,\nx-1.0.dist-info/RECORD,,\n")
    package.inject(wheel, stage)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        record = archive.read("x-1.0.dist-info/RECORD").decode()
        # Files setuptools already packaged are kept as they are
        assert archive.read("openmm_fep/__init__.py") == b"packaged"
    assert "OpenMM.libs/lib/libOpenMMColvars.so" in names
    assert "_openmm_colvars.cpython-312-x86_64-linux-gnu.so" in names
    assert not [n for n in names if "__pycache__" in n]
    assert names.count("openmm_colvars.py") == 1
    assert "OpenMM.libs/lib/libOpenMMColvars.so,sha256=" in record


def test_inject_skips_build_metadata(tmp_path):
    # setuptools writes its egg-info into the staging tree, which is the base wheel's package dir
    stage = tmp_path / "stage"
    (stage / "pkg.egg-info").mkdir(parents=True)
    (stage / "pkg.egg-info/PKG-INFO").write_text("")
    (stage / "openmm_colvars.py").write_text("")
    wheel = tmp_path / "x-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("x-1.0.dist-info/RECORD", "x-1.0.dist-info/RECORD,,\n")
    package.inject(wheel, stage)
    with zipfile.ZipFile(wheel) as archive:
        assert not [n for n in archive.namelist() if ".egg-info" in n]


def test_arrange_orders_by_post_and_kind(tmp_path):
    source, destination = tmp_path / "in", tmp_path / "out"
    (source / "a").mkdir(parents=True)
    names = [
        "openmm_colvars-0.3.0-cp312-cp312-manylinux_2_34_x86_64.whl",
        "openmm_colvars-0.3.0.post2-cp312-cp312-manylinux_2_34_x86_64.whl",
        "openmm_colvars_cuda_12-0.3.0.post2-py3-none-manylinux_2_34_x86_64.whl",
    ]
    for name in names:
        (source / "a" / name).write_bytes(b"")
    package.arrange(source, destination, expected=None)
    assert sorted(p.name for p in (destination / "0-base").iterdir()) == [names[0]]
    assert sorted(p.name for p in (destination / "2-base").iterdir()) == [names[1]]
    # Each CUDA add-on is its own PyPI project, published from its own environment
    assert sorted(p.name for p in (destination / "2-cuda-12").iterdir()) == [names[2]]


def test_arrange_checks_count(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "openmm_colvars-0.3.0-cp312-cp312-manylinux_2_34_x86_64.whl").write_bytes(b"")
    with pytest.raises(SystemExit, match="expected 2"):
        package.arrange(tmp_path / "in", tmp_path / "out", expected=2)


OTOOL = """/tmp/x/OpenMM.libs/lib/libOpenMMColvars.dylib:
\t@rpath/libOpenMMColvars.dylib (compatibility version 0.0.0, current version 0.0.0)
\t@rpath/libOpenMM.dylib (compatibility version 0.0.0, current version 0.0.0)
\t/System/Library/Frameworks/OpenCL.framework/Versions/A/OpenCL (compatibility version 1.0.0)
\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1800.101.0)
\t/Users/runner/work/_temp/build-env/lib/python3.12/site-packages/OpenMM.libs/lib/libOpenMMOpenCL.dylib (x)
\t/opt/homebrew/lib/libfoo.dylib (compatibility version 1.0.0)
"""


def test_foreign_macos_dependencies():
    # Anything not resolved through rpaths or the system would break on users' machines
    assert package.foreign_dependencies(OTOOL) == [
        "/Users/runner/work/_temp/build-env/lib/python3.12/site-packages/OpenMM.libs/lib/libOpenMMOpenCL.dylib",
        "/opt/homebrew/lib/libfoo.dylib",
    ]
