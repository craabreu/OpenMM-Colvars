"""Tests for ColvarsCollectiveVariable, a serializable force whose energy is the value
of a Colvars collective variable."""

import copy
import math
import pickle

import numpy as np
import openmm
import pytest
from openmm_colvars import ColvarsCollectiveVariable, ColvarsForce

PLATFORMS = ["Reference", "CPU"]
RADIANS_TO_DEGREES = 180 / math.pi
NM_TO_ANGSTROM = 10.0


def make_positions(num_particles, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.3, size=(num_particles, 3))


def make_system(num_particles, *forces):
    system = openmm.System()
    for _ in range(num_particles):
        system.addParticle(12.0)
    for group, force in enumerate(forces):
        force.setForceGroup(group)
        system.addForce(force)
    return system


def make_context(system, positions, platform="Reference"):
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName(platform),
    )
    context.setPositions(positions)
    return context


def energy_and_forces(context, group):
    state = context.getState(getEnergy=True, getForces=True, groups={group})
    energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        openmm.unit.kilojoule_per_mole / openmm.unit.nanometer
    )
    return energy, forces


def colvar_config(name, component, extra=""):
    return f"colvar {{\n  name {name}\n{extra}  {component}\n}}\n"


def atoms(*indices):
    return " ".join(str(i + 1) for i in indices)


DISTANCE = (
    "distance {\n    group1 { atomNumbers "
    + atoms(0)
    + " }\n    group2 { atomNumbers "
    + atoms(1)
    + " }\n  }"
)
DIHEDRAL = (
    "dihedral {\n"
    + "".join(f"    group{k + 1} {{ atomNumbers {atoms(k)} }}\n" for k in range(4))
    + "  }"
)

# A customFunction converts the dihedral from degrees to radians
DIHEDRAL_IN_RADIANS = colvar_config(
    "phi",
    DIHEDRAL.replace("dihedral {\n", "dihedral {\n    name t\n", 1),
    "  customFunction t*0.017453292519943295\n",
)


def test_distance_value_and_forces():
    # Colvars reports distances in Angstrom
    cv = ColvarsCollectiveVariable(colvar_config("d", DISTANCE), "d")
    reference = openmm.CustomBondForce(f"{NM_TO_ANGSTROM}*r")
    reference.addBond(0, 1, [])
    positions = make_positions(2)
    context = make_context(make_system(2, cv, reference), positions)
    energy, forces = energy_and_forces(context, 0)
    expected_energy, expected_forces = energy_and_forces(context, 1)
    assert energy == pytest.approx(expected_energy, rel=1e-6)
    assert forces == pytest.approx(expected_forces, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_periodic_dihedral_value_and_forces(seed):
    # Colvars reports dihedrals in degrees, wrapped into (-180, 180]; this is the
    # same convention as OpenMM's theta in radians
    cv = ColvarsCollectiveVariable(colvar_config("phi", DIHEDRAL), "phi")
    reference = openmm.CustomTorsionForce(f"{RADIANS_TO_DEGREES}*theta")
    reference.addTorsion(0, 1, 2, 3, [])
    positions = make_positions(4, seed)
    context = make_context(make_system(4, cv, reference), positions)
    energy, forces = energy_and_forces(context, 0)
    expected_energy, expected_forces = energy_and_forces(context, 1)
    assert energy == pytest.approx(expected_energy, rel=1e-6)
    assert forces == pytest.approx(expected_forces, rel=1e-5, abs=1e-6)


def rotation_matrix(seed):
    rng = np.random.default_rng(seed)
    w, x, y, z = (q := rng.normal(size=4)) / np.linalg.norm(q)
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return matrix, 2 * math.degrees(math.acos(min(1.0, abs(w))))


def orientation_angle(reference):
    ref_positions = "\n".join(
        f"      ({x * NM_TO_ANGSTROM:.10f}, {y * NM_TO_ANGSTROM:.10f}, "
        f"{z * NM_TO_ANGSTROM:.10f})"
        for x, y, z in reference
    )
    component = (
        "orientationAngle {\n    atoms { atomNumbers "
        + atoms(*range(len(reference)))
        + " }\n    refPositions {\n"
        + ref_positions
        + "\n    }\n  }"
    )
    return colvar_config("omega", component)


def test_orientation_angle_matches_applied_rotation():
    reference = make_positions(6)
    rotation, angle = rotation_matrix(1)
    center = reference.mean(axis=0)
    positions = (reference - center) @ rotation.T + center
    cv = ColvarsCollectiveVariable(orientation_angle(reference), "omega")
    context = make_context(make_system(6, cv), positions)
    energy, _ = energy_and_forces(context, 0)
    assert energy == pytest.approx(angle, abs=1e-6)


def test_getters_and_validation():
    cv = ColvarsCollectiveVariable(colvar_config("d", DISTANCE), "d")
    assert cv.getConfigString() == colvar_config("d", DISTANCE)
    assert cv.getColvarName() == "d"
    assert not cv.usesPeriodicBoundaryConditions()
    cv.setUsesPeriodicBoundaryConditions(True)
    assert cv.usesPeriodicBoundaryConditions()
    with pytest.raises(openmm.OpenMMException, match="configString"):
        ColvarsCollectiveVariable("", "d")
    with pytest.raises(openmm.OpenMMException, match="colvarName"):
        ColvarsCollectiveVariable(colvar_config("d", DISTANCE), "")


def assert_same_settings(copy_, original):
    assert isinstance(copy_, ColvarsCollectiveVariable)
    assert copy_.getConfigString() == original.getConfigString()
    assert copy_.getColvarName() == original.getColvarName()
    assert copy_.usesPeriodicBoundaryConditions() == original.usesPeriodicBoundaryConditions()
    assert copy_.getForceGroup() == original.getForceGroup()
    assert copy_.getName() == original.getName()


def configured_cv():
    cv = ColvarsCollectiveVariable(colvar_config("phi", DIHEDRAL), "phi")
    cv.setForceGroup(7)
    cv.setName("phi_colvars")
    cv.setUsesPeriodicBoundaryConditions(True)
    return cv


def test_serialization_round_trip():
    cv = configured_cv()
    xml = openmm.XmlSerializer.serialize(cv)
    assert "ColvarsCollectiveVariable" in xml
    # OpenMM returns plugin forces as plain Force objects, which work in a System
    restored = openmm.XmlSerializer.deserialize(xml)
    assert openmm.XmlSerializer.serialize(restored) == xml
    assert_same_settings(ColvarsCollectiveVariable._deserialize(xml), cv)
    positions = make_positions(4, 2)
    restored.setForceGroup(0)
    energy, forces = energy_and_forces(make_context(make_system(4, restored), positions), 0)
    expected = make_context(make_system(4, configured_cv()), positions)
    expected_energy, expected_forces = energy_and_forces(expected, 0)
    assert energy == pytest.approx(expected_energy)
    assert forces == pytest.approx(expected_forces)


@pytest.mark.parametrize(
    "duplicate",
    [copy.copy, copy.deepcopy, lambda cv: pickle.loads(pickle.dumps(cv))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_python_copies(duplicate):
    cv = configured_cv()
    assert_same_settings(duplicate(cv), cv)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_inside_a_custom_cv_force(platform):
    # A periodic harmonic coupling of a Colvars dihedral in radians with an OpenMM
    # distance
    phi = ColvarsCollectiveVariable(DIHEDRAL_IN_RADIANS, "phi")
    distance = openmm.CustomBondForce("r")
    distance.addBond(0, 3, [])
    expression = "0.5*k*dphi^2 + d; dphi = phi - phi0 - 2*pi*floor((phi - phi0)/(2*pi) + 0.5)"
    coupling = openmm.CustomCVForce(expression.replace("pi", str(math.pi)))
    coupling.addGlobalParameter("k", 100.0)
    coupling.addGlobalParameter("phi0", 2.5)
    coupling.addCollectiveVariable("phi", phi)
    coupling.addCollectiveVariable("d", distance)

    torsion = openmm.CustomTorsionForce("theta")
    torsion.addTorsion(0, 1, 2, 3, [])
    expected = openmm.CustomCVForce(expression.replace("pi", str(math.pi)))
    expected.addGlobalParameter("k", 100.0)
    expected.addGlobalParameter("phi0", 2.5)
    expected.addCollectiveVariable("phi", torsion)
    expected.addCollectiveVariable("d", copy.deepcopy(distance))

    positions = make_positions(4, 3)
    context = make_context(make_system(4, coupling, expected), positions, platform)
    energy, forces = energy_and_forces(context, 0)
    expected_energy, expected_forces = energy_and_forces(context, 1)
    assert energy == pytest.approx(expected_energy, rel=1e-6)
    assert forces == pytest.approx(expected_forces, rel=1e-5, abs=1e-6)
    values = coupling.getCollectiveVariableValues(context)
    assert values == pytest.approx(expected.getCollectiveVariableValues(context))


def test_several_collective_variables_in_one_force():
    # Every instance owns its own Colvars module, so two of them must not interfere
    def torsion_cv(name, first):
        groups = "".join(
            f"    group{k + 1} {{ atomNumbers {atoms(first + k)} }}\n" for k in range(4)
        )
        component = "dihedral {\n    name t\n" + groups + "  }"
        config = colvar_config(name, component, "  customFunction t*0.017453292519943295\n")
        return ColvarsCollectiveVariable(config, name)

    def torsion_reference(first):
        force = openmm.CustomTorsionForce("theta")
        force.addTorsion(first, first + 1, first + 2, first + 3, [])
        return force

    expression = "phi^2 + 2*psi^2"
    colvars = openmm.CustomCVForce(expression)
    expected = openmm.CustomCVForce(expression)
    for name, first in [("phi", 0), ("psi", 1)]:
        colvars.addCollectiveVariable(name, torsion_cv(name, first))
        expected.addCollectiveVariable(name, torsion_reference(first))
    positions = make_positions(5, 4)
    context = make_context(make_system(5, colvars, expected), positions)
    energy, forces = energy_and_forces(context, 0)
    expected_energy, expected_forces = energy_and_forces(context, 1)
    assert energy == pytest.approx(expected_energy, rel=1e-6)
    assert forces == pytest.approx(expected_forces, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize(
    "config, name, message",
    [
        (colvar_config("d", DISTANCE), "missing", "does not define a colvar"),
        (
            colvar_config("d", DISTANCE)
            + "harmonic {\n  colvars d\n  centers 1.0\n  forceConstant 1.0\n}\n",
            "d",
            "must not define biases",
        ),
        (
            colvar_config(
                "d",
                DISTANCE,
                "  extendedLagrangian on\n  extendedFluctuation 0.1\n"
                "  extendedTimeConstant 200\n",
            ),
            "d",
            "extended Lagrangian",
        ),
        (
            colvar_config("v", DISTANCE.replace("distance {", "distanceVec {")),
            "v",
            "not a scalar",
        ),
        (
            "scriptedColvarForces off\n" + colvar_config("d", DISTANCE),
            "d",
            "scriptedColvarForces",
        ),
    ],
    ids=["unknown-colvar", "bias", "extended-Lagrangian", "vector", "scripted-forces-off"],
)
def test_invalid_configurations_are_rejected(config, name, message):
    cv = ColvarsCollectiveVariable(config, name)
    with pytest.raises(openmm.OpenMMException, match=message):
        make_context(make_system(2, cv), make_positions(2))


def test_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cv = ColvarsCollectiveVariable(
        colvar_config("phi", DIHEDRAL, "  # a comment\n"), "phi"
    )
    system = make_system(4, cv)
    context = make_context(system, make_positions(4))
    integrator = context.getIntegrator()
    integrator.step(300)
    del context
    assert list(tmp_path.iterdir()) == []


def test_colvars_force_is_unchanged(tmp_path):
    config = colvar_config("d", DISTANCE) + (
        "harmonic {\n  name restraint\n  colvars d\n  centers 3.0\n"
        "  forceConstant 1.0\n}\n"
    )
    force = ColvarsForce.fromConfigString(config, str(tmp_path / "colvars"))
    positions = [openmm.Vec3(0, 0, 0), openmm.Vec3(0.35, 0, 0)]
    context = make_context(make_system(2, force), positions)
    energy, _ = energy_and_forces(context, 0)
    # 0.5 * 1 kcal/mol/A^2 * (3.5 A - 3.0 A)^2 = 0.125 kcal/mol
    assert energy == pytest.approx(0.125 * 4.184)
