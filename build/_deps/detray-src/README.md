# detray

Detray is part of the ACTS project (R&D line for parallelization), the ACTS project can be found: https://github.com/acts-project/acts.

This is a C++20 header only library for detector surface intersections using different algebra plugin libraries. It follows the navigation and propagation concept of ACTS, however, with an attempt to create
a geometry without polymorphic inheritance structure.


## Requirements and dependencies
#### OS & compilers:

- The C++ compiler must support C++20
- The CUDA Toolkit version must be greater than major version 11

#### Dependency
- CMake

## Getting started

The respository is meant to be possible to build "out of the box", with standard
CMake build procedures.

```shell
git clone https://github.com/acts-project/detray.git
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -S detray -B detray-build
cmake --build detray-build
```

For tests and benchmarks with the inhomogeneous magnetic field, the ODD field in covfie format should be downloaded and the environment variable should be set.
```shell
cd detray/data
bash detray_data_get_files.sh
export DETRAY_BFIELD_FILE="${PWD}/odd-bfield_v0_9_0.cvf"
```

#### Build options

A number of cmake preset configurations are provided and can be listed by:
```shell
cmake -S detray --list-presets
```
For a developer build, the ```dev-fp32``` and ```dev-fp64``` configurations are available (```fp```: floating point precision):
```shell
cmake -S detray -B detray-build --preset dev-fp32
```
The developer presets will fetch all dependencies, but not automatically trigger the build of additional detray components. For example, in order to trigger the build of the unit tests, the corresponding option needs to be specified:
```shell
cmake -S detray -B detray-build --preset dev-fp32 \
-DDETRAY_BUILD_UNITTESTS=ON
```
A full build, containing all components (e.g. tests and benchmarks), can be configured using the ```full-fp32``` and ```full-fp64``` presets.

The following cmake options are available and can also be specified explicitly for any preset:

| Option | Description | Default |
| --- | --- | --- |
| DETRAY_BUILD_CUDA  | Build the CUDA sources included in detray | ON (if available) |
| DETRAY_BUILD_SYCL  | Build the SYCL sources included in detray | OFF |
| DETRAY_BUILD_TEST_UTILS  | Build the detray test utilities library (contains e.g. test detectors) | OFF |
| DETRAY_BUILD_UNITTESTS  | Build the detray unit tests | OFF |
| DETRAY_BUILD_INTEGRATIONTESTS  | Build the detray integration tests | OFF |
| DETRAY_BUILD_ALL_TESTS  | Build the detray unit and integration tests | OFF |
| DETRAY_BUILD_BENCHMARKS  | Build the detray benchmarks | OFF |
| DETRAY_BUILD_CLI_TOOLS  | Build the detray command line tools | OFF |
| DETRAY_BUILD_TUTORIALS  | Build the examples of detray | OFF |
| DETRAY_CUSTOM_SCALARTYPE | Floating point precision | double |
| DETRAY_EIGEN_PLUGIN | Build Eigen math plugin | OFF |
| DETRAY_SMATRIX_PLUGIN | Build ROOT/SMatrix math plugin | OFF |
| DETRAY_VC_AOS_PLUGIN | Build Vc based AoS math plugin | OFF |
| DETRAY_VC_SOA_PLUGIN | Build Vc based SoA math plugin (currently only supports the ray-surface intersectors) | OFF |
| DETRAY_SVG_DISPLAY | Build ActSVG display module | OFF |

## Continuous benchmark

Monitoring the propagation speed with the toy geometry

<img src="https://gitlab.cern.ch/acts/detray-benchmark/-/raw/master/plots/array_data.png?ref_type=heads" />
