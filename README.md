# traccc

Demonstrator tracking chain for accelerators.

## Features

| Category           | Algorithms             | CPU | CUDA | SYCL |
| ------------------ | ---------------------- | --- | ---- | ---- |
| **Clusterization** | CCL                    | ✅  | 🟡   | 🟡   |
|                    | Measurement creation   | ✅  | 🟡   | 🟡   |
|                    | Spacepoint formation   | ✅  | 🟡   | 🟡   |
| **Track finding**  | Spacepoint binning     | ✅  | 🟡   | ⚪   |
|                    | Seed finding           | ✅  | ✅   | 🟡   |
|                    | Track param estimation | ✅  | 🟡   | ⚪   |
|                    | Combinatorial KF       | ⚪  | ⚪   | ⚪   |
| **Track fitting**  | KF                     | 🟡  | 🟡   | ⚪   |

✅: exists, 🟡: work started, ⚪: work not started yet

## Requirements and dependencies 

#### OS & compilers:
- gcc should support c++17
- Following table lists the (currently idenfitifed) working combinations of OS and compilers.

| OS | gcc | cuda | comment |
| --- | --- | --- | --- |
| Ubuntu 20.04   | 9.3.0 | 11.3 | runs on CI |
| Centos 8   | 8.4.1 | 11.3 | |

#### Data directory
- the `data` directory is a submodule hosted as `git lfs` on `https://gitlab.cern.ch/acts/traccc-data`

## Getting started

### Clone the repository

Clone the repository and setup up the submodules, this requires `git-lfs` for the data from the `traccc-data` repository.

```sh
git clone git@github.com:acts-project/traccc.git
cd traccc
git submodule update --init
```

### Build the project

```sh
cmake -S . -B <build_directory>
cmake --build <build_directory> <options>
```

### Build options

| Option | Description | 
| --- | --- |
| TRACCC_BUILD_CUDA  | Build the CUDA sources included in traccc |
| TRACCC_BUILD_SYCL  | Build the SYCL sources included in traccc |
| TRACCC_BUILD_TESTING  | Build the (unit) tests of traccc |
| TRACCC_USE_SYSTEM_VECMEM | Pick up an existing installation of VecMem from the build environment |
| TRACCC_USE_SYSTEM_EIGEN3 | Pick up an existing installation of Eigen3 from the build environment |
| TRACCC_USE_SYSTEM_ALGEBRA_PLUGINS | Pick up an existing installation of Algebra Plugins from the build environment |
| TRACCC_USE_SYSTEM_DFELIBS | Pick up an existing installation of dfelibs from the build environment |
| TRACCC_USE_SYSTEM_DETRAY | Pick up an existing installation of Detray from the build environment |
| TRACCC_USE_SYSTEM_ACTS | Pick up an existing installation of Acts from the build environment |
| TRACCC_USE_SYSTEM_GOOGLETEST | Pick up an existing installation of GoogleTest from the build environment |

## Examples

### cpu reconstruction chain

```sh
<build_directory>/bin/seq_example tml_detector/trackml-detector.csv tml_pixels/ <number of events> 
```

### cuda reconstruction chain

- Users can generate cuda examples by adding `-DTRACCC_BUILD_CUDA=ON` to cmake options

```sh
<build_directory>/bin/seq_example_cuda tml_detector/trackml-detector.csv tml_pixels/ <number of events> <run cpu tracking>
```
