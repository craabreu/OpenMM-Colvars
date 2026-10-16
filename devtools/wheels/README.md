# PyPI wheels

The [PyPI workflow](../../.github/workflows/PyPI.yml) builds and publishes three projects:

| Project | Contents | Platforms |
|---|---|---|
| `openmm-colvars` | `openmm_colvars`, `openmm_fep` and `libOpenMMColvars` | Linux x86_64, macOS arm64 and x86_64 |
| `openmm-colvars-cuda-12` | the CUDA plugin, `libOpenMMColvarsCUDA`, for CUDA 12 | Linux x86_64 |
| `openmm-colvars-cuda-13` | the CUDA plugin for CUDA 13 | Linux x86_64 |

Users install them with `pip install openmm-colvars` or `pip install "openmm-colvars[cuda12]"`.
The base wheels are built with `-DOPENMM_COLVARS_CUDA_PLUGIN=ON`, so they work on machines
without CUDA. Without the CUDA plugin, ordinary biases still run on the CUDA platform, but
ABF/eABF and `AlchemicalPMEForce` do not. All libraries go into `site-packages/OpenMM.libs`,
next to those of pip's OpenMM, which loads the plugin when it is imported.

## Versions

The C++ code uses OpenMM internals, so each source release `X.Y.Z` (the `project()` version in
`CMakeLists.txt`) is published once for each OpenMM minor version in
[openmm-versions.json](openmm-versions.json), as `X.Y.Z`, `X.Y.Z.post1`, `X.Y.Z.post2`, ...
in increasing order of OpenMM version. Each one requires its OpenMM minor version, so pip picks
the release that matches the OpenMM in the environment. To support a new OpenMM version, add
an entry with the next post number.

The workflow builds against the Colvars commit in `COLVARS_COMMIT`, near the top of
`PyPI.yml`. Later Colvars commits changed the proxy API that `OpenMMColvarProxy` uses.

## One-time setup

Publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API
token is stored in the repository.

1. On GitHub, under *Settings → Environments*, create the environments `pypi`, `pypi-cuda-12`
   and `pypi-cuda-13`, each with at least one required reviewer. The workflow refuses to
   publish to PyPI otherwise.
2. On [pypi.org](https://pypi.org/manage/account/publishing/), add a pending trusted publisher
   for each project, with this repository, workflow `PyPI.yml` and the environment named
   after the project:

   | PyPI project | Environment |
   |---|---|
   | `openmm-colvars` | `pypi` |
   | `openmm-colvars-cuda-12` | `pypi-cuda-12` |
   | `openmm-colvars-cuda-13` | `pypi-cuda-13` |

For [TestPyPI](https://test.pypi.org/manage/account/publishing/), do the same with the
environments `testpypi`, `testpypi-cuda-12` and `testpypi-cuda-13`, which need no reviewers.

## Releasing

1. Set the new version in the `project()` call of `CMakeLists.txt`.
2. Optionally, test the release on TestPyPI: run the workflow manually (*Actions → PyPI →
   Run workflow*) with the commit and target `testpypi`. After publishing, the workflow checks
   that pip resolves each OpenMM version to its release.
3. Publish a GitHub release whose tag is the version, with or without a leading `v`. The
   workflow checks the tag against `CMakeLists.txt`, builds every wheel, and waits for a
   reviewer to approve the upload to PyPI. Prereleases are not published.

A manual run with target `none` builds and tests every wheel without publishing them.

## Local checks

```bash
pip install pytest setuptools wheel delocate
python -m pytest devtools/wheels/tests
```
