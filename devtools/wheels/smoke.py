"""Checks run in a clean venv after installing the wheels."""

import argparse
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package import link_problems  # noqa: E402

# A harmonic restraint on the distance between two particles, 15 A apart: Colvars works
# in A and kcal/mol, so the energy is 0.5*1*(15-10)^2 kcal/mol = 52.3 kJ/mol
RESTRAINT = """
colvarsTrajFrequency 0
colvar {
  name d
  distance {
    group1 { atomNumbers 1 }
    group2 { atomNumbers 2 }
  }
}
harmonic {
  colvars d
  centers 10.0
  forceConstant 1.0
}
"""
# ABF needs the total force on the colvar, which on CUDA only the plugin provides
ABF = """
colvarsTrajFrequency 0
colvar {
  name d
  width 0.1
  lowerBoundary 1.0
  upperBoundary 10.0
  distance {
    group1 { atomNumbers 1 }
    group2 { atomNumbers 2 }
  }
}
abf {
  colvars d
  fullSamples 1000
}
"""


def _context(config, platform_name, physical_force=None):
    import openmm
    from openmm_colvars import ColvarsForce

    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0)
    if physical_force is not None:
        system.addForce(physical_force)
    system.addForce(ColvarsForce.fromConfigString(config, f"smoke_{platform_name}"))
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(1.5, 0, 0)])
    return context


def _restraint_energy(platform_name):
    energy = _context(RESTRAINT, platform_name).getState(getEnergy=True).getPotentialEnergy()
    print(f"{platform_name} restraint energy: {energy}")
    assert abs(energy._value - 52.3) < 1e-4, energy


def check_base(expected):
    import openmm
    import openmm_colvars  # noqa: F401
    import openmm_fep  # noqa: F401

    failures = [f for f in openmm.Platform.getPluginLoadFailures()
                if "Colvars" in f and "ColvarsCUDA" not in f]
    assert not failures, f"plugin load failures: {failures}"
    assert version("openmm-colvars") == expected, version("openmm-colvars")
    _restraint_energy("Reference")


def check_cuda(expected, cuda):
    import openmm.version

    assert version(f"openmm-colvars-cuda-{cuda}") == expected
    plugin = Path(openmm.version.openmm_library_path) / "plugins" / "libOpenMMColvarsCUDA.so"
    # Relocating every symbol catches a plugin built against newer CUDA libraries than pip installed
    result = subprocess.run(["ldd", "-r", str(plugin)], capture_output=True, text=True)
    output = result.stdout + result.stderr
    problems = link_problems(output)
    assert not problems, "\n".join(problems) + f"\n{output}"


def check_gpu():
    # Needs an NVIDIA GPU, so it runs only on local machines
    import openmm

    failures = [f for f in openmm.Platform.getPluginLoadFailures() if "Colvars" in f]
    assert not failures, f"plugin load failures: {failures}"
    platform = openmm.Platform.getPlatformByName("CUDA")
    assert platform.supportsKernels(["InitializeDeferredColvars"]), "the CUDA plugin was not loaded"
    _restraint_energy("CUDA")
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.25, 1000.0)
    context = _context(ABF, "CUDA", bond)
    context.getIntegrator().step(10)
    print("ABF on CUDA: OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["base", "cuda", "gpu"])
    parser.add_argument("expected", nargs="?")
    parser.add_argument("cuda", nargs="?")
    args = parser.parse_args()
    # Colvars writes output files into the working directory
    os.chdir(tempfile.mkdtemp())
    if args.kind == "base":
        check_base(args.expected)
    elif args.kind == "cuda":
        check_cuda(args.expected, args.cuda)
    else:
        check_gpu()
    print("OK")


if __name__ == "__main__":
    main()
