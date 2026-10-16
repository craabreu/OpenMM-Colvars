import email
import os
import subprocess
import sys
import zipfile
from pathlib import Path

WHEELS = Path(__file__).parents[1]


def _source(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("PROJECT(OpenMMColvars VERSION 0.3.0)\n")
    return source


def _stage(tmp_path):
    stage = tmp_path / "stage"
    (stage / "openmm_fep").mkdir(parents=True)
    (stage / "openmm_colvars.py").write_text("")
    (stage / "openmm_fep/__init__.py").write_text("")
    return stage


def _build(project, out, env):
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(out), str(project)],
        check=True, env={**os.environ, **env}, capture_output=True)
    (wheel,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        (metadata,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        return wheel, email.message_from_bytes(archive.read(metadata))


def test_base_wheel_metadata(tmp_path):
    stage = _stage(tmp_path)
    wheel, meta = _build(WHEELS, tmp_path / "out", {
        "OC_SOURCE_DIR": str(_source(tmp_path)), "OC_OPENMM_MINOR": "8.5", "OC_STAGE_DIR": str(stage)})
    assert meta["Name"] == "openmm-colvars"
    assert meta["Version"] == "0.3.0.post1"
    requires = meta.get_all("Requires-Dist")
    assert "openmm<8.6,>=8.5.2" in requires
    assert "-none-any" not in wheel.name
    with zipfile.ZipFile(wheel) as archive:
        assert {"openmm_colvars.py", "openmm_fep/__init__.py"} <= set(archive.namelist())


def test_each_build_packages_its_own_staging_tree(tmp_path):
    source = _source(tmp_path)
    for content in ("first", "second"):
        stage = tmp_path / content
        (stage / "openmm_fep").mkdir(parents=True)
        (stage / "openmm_colvars.py").write_text(content)
        (stage / "openmm_fep/__init__.py").write_text(content)
        wheel, _ = _build(WHEELS, tmp_path / f"out-{content}", {
            "OC_SOURCE_DIR": str(source), "OC_OPENMM_MINOR": "8.4", "OC_STAGE_DIR": str(stage)})
        with zipfile.ZipFile(wheel) as archive:
            assert archive.read("openmm_fep/__init__.py").decode() == content
    assert not (WHEELS / "build").exists()


def test_cuda_extras_are_linux_only(tmp_path):
    stage = _stage(tmp_path)
    _, meta = _build(WHEELS, tmp_path / "out", {
        "OC_SOURCE_DIR": str(_source(tmp_path)), "OC_OPENMM_MINOR": "8.6", "OC_STAGE_DIR": str(stage)})
    requires = meta.get_all("Requires-Dist")
    assert ('openmm-colvars-cuda-12==0.3.0.post2; platform_system == "Linux" and extra == "cuda12"'
            in requires)
    assert set(meta.get_all("Provides-Extra")) == {"cuda12", "cuda13"}


def test_cuda_wheel_metadata(tmp_path):
    wheel, meta = _build(WHEELS / "cuda", tmp_path / "out", {
        "OC_SOURCE_DIR": str(_source(tmp_path)), "OC_OPENMM_MINOR": "8.4", "OC_CUDA": "13",
        "OC_PLATFORM": "manylinux_2_34_x86_64"})
    assert meta["Name"] == "openmm-colvars-cuda-13"
    assert meta["Version"] == "0.3.0"
    assert set(meta.get_all("Requires-Dist")) == {
        "openmm-cuda-13<8.5,>=8.4.0.post2", "openmm-colvars==0.3.0"}
    assert wheel.name.endswith("-py3-none-manylinux_2_34_x86_64.whl")


def test_description_comes_from_the_built_source(tmp_path):
    # The project page must describe the release being built, not the tooling checkout
    source = _source(tmp_path)
    (source / "README.md").write_text("Release README\n")
    stage = _stage(tmp_path)
    _, meta = _build(WHEELS, tmp_path / "out", {
        "OC_SOURCE_DIR": str(source), "OC_OPENMM_MINOR": "8.4", "OC_STAGE_DIR": str(stage)})
    assert meta.get_payload().strip() == "Release README"
