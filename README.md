# traccc

Demonstrator tracking chain for accelerators.

## Features

| Category           | Algorithms             | CPU | CUDA | SYCL | Futhark |
| ------------------ | ---------------------- | --- | ---- | ---- | ------- |
| **Clusterization** | CCL                    | ✅  | 🟡   | 🟡   | ✅      |
|                    | Measurement creation   | ✅  | 🟡   | 🟡   | ✅      |
|                    | Spacepoint formation   | ✅  | 🟡   | 🟡   | ⚪      |
| **Track finding**  | Spacepoint binning     | ✅  | ✅   | ✅   | ⚪      |
|                    | Seed finding           | ✅  | ✅   | ✅   | ⚪      |
|                    | Track param estimation | ✅  | ✅   | ✅   | ⚪      |
|                    | Combinatorial KF       | ⚪  | ⚪   | ⚪   | ⚪      |
| **Track fitting**  | KF                     | 🟡  | 🟡   | ⚪   | ⚪      |

✅: exists, 🟡: work started, ⚪: work not started yet

The relations between datatypes and algorithms is given in the (approximately
commutative) diagram shown below. Black lines indicate CPU algorithms, green
lines indicate CUDA algorithms, blue lines indicate SYCL algorithms, and brown
lines indicate Futhark algorithms. Solid algorithms are ready for use, dashed
algorithms are in development or future goals. Data types for different
heterogeneous platforms are contracted for legibility, and identities are
hidden.

```mermaid
flowchart LR
    subgraph clusterization [<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/clusterization/clusterization_algorithm.hpp'>Clusterization</a>]
        direction TB
        cell(Cells);
        cluster(Clusters);
        meas(Measurements);
    end

    subgraph trkfinding [Track Finding]
        sp(Spacepoints);
        bin(Bins);
        seed(Seeds);
        ptrack(Prototracks);
    end

    subgraph trkfitting [Track Fitting]
        track(Track);
    end

    click cell href "https://github.com/acts-project/traccc/blob/main/core/include/traccc/edm/cell.hpp";
    click cluster href "https://github.com/acts-project/traccc/blob/main/core/include/traccc/edm/cluster.hpp";
    click meas href "https://github.com/acts-project/traccc/blob/main/core/include/traccc/edm/measurement.hpp";
    click sp href "https://github.com/acts-project/traccc/blob/main/core/include/traccc/edm/spacepoint.hpp";
    click seed href "https://github.com/acts-project/traccc/blob/main/core/include/traccc/edm/seed.hpp";
    click ptrack href "https://github.com/acts-project/traccc/blob/main/core/include/traccc/edm/track_parameters.hpp";

    %% CPU CCL algorithm
    cell -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/clusterization/component_connection.hpp'>CCL</a>| cluster;
    linkStyle 0 stroke: black;

    %% SYCL CCL algorithm
    cell -.->|CCL| cluster;
    linkStyle 1 stroke: blue;

    %% CPU clusterization
    cluster -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/clusterization/measurement_creation.hpp'>Agg.</a>| meas;
    linkStyle 2 stroke: black;

    %% SYCL clusterization
    cluster -.->|Agg.| meas;
    linkStyle 3 stroke: blue;

    %% CUDA CCA
    cell -.->|CCA| meas;
    linkStyle 4 stroke: green;

    %% CPU local to global
    meas -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/clusterization/spacepoint_formation.hpp'>L2G</a>| sp;
    linkStyle 5 stroke: black;

    %% SYCL local to global
    meas -.->|L2G| sp;
    linkStyle 6 stroke: blue;

    %% CUDA local to global
    meas -.->|L2G| sp;
    linkStyle 7 stroke: green;

    %% CPU binning
    sp -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/seeding/spacepoint_binning.hpp'>Binning</a>| bin;
    linkStyle 8 stroke: black;

    %% CUDA binning
    sp -->|<a href='https://github.com/acts-project/traccc/blob/main/device/cuda/include/traccc/cuda/seeding/spacepoint_binning.hpp'>Binning</a>| bin;
    linkStyle 9 stroke: green;

    %% CPU seeding
    bin -.->|Seeding| seed;
    linkStyle 10 stroke: black;

    %% SYCL seeding
    bin -->|<a href='https://github.com/acts-project/traccc/blob/main/device/sycl/include/traccc/sycl/seeding/seed_finding.hpp'>Seeding</a>| seed;
    linkStyle 11 stroke: blue;

    %% CUDA seeding
    bin -->|<a href='https://github.com/acts-project/traccc/tree/main/device/cuda/include/traccc/cuda/seeding'>Seeding</a>| seed;
    linkStyle 12 stroke: green;

    %% CUDA binless seeding
    sp -.->|Seeding| seed;
    linkStyle 13 stroke: green;

    %% CPU param est.
    seed -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/seeding/track_params_estimation.hpp'>Param. Est.</a>| ptrack;
    linkStyle 14 stroke: black;

    %% CUDA param est.
    seed -->|<a href='https://github.com/acts-project/traccc/blob/main/device/cuda/include/traccc/cuda/seeding/track_params_estimation.hpp'>Param. Est.</a>| ptrack;
    linkStyle 15 stroke: green;

    %% CPU CKF
    ptrack -.->|CKF| track;
    linkStyle 16 stroke: black;

    %% CPU Kalman filter
    track -.->|Kalman filter| track;
    linkStyle 17 stroke: black;

    %% CUDA kalman filter
    track -.->|Kalman filter| track;
    linkStyle 18 stroke: green;

    %% SYCL binning
    sp -->|<a href='https://github.com/acts-project/traccc/blob/main/device/sycl/include/traccc/sycl/seeding/spacepoint_binning.hpp'>Binning</a>| bin;
    linkStyle 19 stroke: blue;

    %% SYCL track parameter est.
    seed -->|<a href='https://github.com/acts-project/traccc/blob/main/device/sycl/include/traccc/sycl/seeding/track_params_estimation.hpp'>Param. Est.</a>| ptrack;
    linkStyle 20 stroke: blue;

    %% Futhark measurement creation
    cell -->|<a href='https://github.com/acts-project/traccc/blob/main/device/futhark/src/measurement_creation.fut'>CCA</a>| meas;
    linkStyle 21 stroke: brown;
```

## Requirements and dependencies 

### OS & compilers:

Please note that due to the complexity of this software and its build system,
it may be somewhat fragile in the face of compiler version changes. The
following are general guidelines for getting _traccc_ to compile:

- The C++ compiler must support C++17

In addition, the following requirements hold when CUDA is enabled:

- The CUDA Toolkit version must be greater than major version 11
- The CUDA Toolkit must not be minor version 11.3 due to a
  [bug](https://github.com/acts-project/traccc/issues/115) in the front-end
  compiler of that version
- Ensure that the CUDA host compiler supports C++17 and is compatible with the
  `nvcc` compiler driver

The following table lists currently combinations of builds, compilers,
and toolchains that are currently known to work (last updated 2022/01/24):

| Build | OS | gcc | cuda | comment |
| --- | --- | --- | --- | --- |
| CUDA | Ubuntu 20.04   | 9.3.0 | 11.5 | runs on CI |

### Data directory

The `data` directory is a submodule hosted as `git lfs` on `https://gitlab.cern.ch/acts/traccc-data`

### Prerequisites

- [Boost](https://www.boost.org/): program_options
- [ROOT](https://root.cern/): RIO, Hist, Tree

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
| TRACCC_BUILD_EXAMPLES  | Build the examples of traccc |
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
<build_directory>/bin/traccc_seq_example --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory=tml_pixels/ --events=10 
```

### cuda reconstruction chain

- Users can generate cuda examples by adding `-DTRACCC_BUILD_CUDA=ON` to cmake options

```sh
<build_directory>/bin/traccc_seq_example_cuda --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory=tml_pixels/ --events=10 --run_cpu=1
```

## Troubleshooting

The following are potentially useful instructions for troubleshooting various
problems with your build:

### CUDA

#### Incompatible host compiler

You may experience errors being issued about standard library features, for example:

```
/usr/include/c++/11/bits/std_function.h:435:145: note:         ‘_ArgTypes’
/usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:
  530 |         operator=(_Functor&& __f)
```

In this case, your `nvcc` host compiler is most likely incompatible with your
CUDA toolkit. Consider installing a supported version and selecting it through
the `CUDAHOSTCXX` environment variable at build-time.
