# traccc

Demonstrator tracking chain for accelerators.

## Features

| Category           | Algorithms             | CPU | CUDA | SYCL | Alpaka | Kokkos | Futhark |
| ------------------ | ---------------------- | --- | ---- | ---- | ------ | ------ | ------- |
| **Clusterization** | CCL / FastSv / etc.    | ✅  | ✅   | ✅   | 🟡     | ⚪     | ✅      |
|                    | Measurement creation   | ✅  | ✅   | ✅   | 🟡     | ⚪     | ✅      |
| **Seeding**        | Spacepoint formation   | ✅  | ✅   | ✅   | 🟡     | ⚪     | ⚪      |
|                    | Spacepoint binning     | ✅  | ✅   | ✅   | ✅     | ✅     | ⚪      |
|                    | Seed finding           | ✅  | ✅   | ✅   | ✅     | ⚪     | ⚪      |
|                    | Track param estimation | ✅  | ✅   | ✅   | ✅     | ⚪     | ⚪      |
| **Track finding**  | Combinatorial KF       | ✅  | ✅   | 🟡   | 🟡     | ⚪     | ⚪      |
| **Track fitting**  | KF                     | ✅  | ✅   | ✅   | ⚪     | ⚪     | ⚪      |
| **Ambiguity resolution**  | Greedy resolver   | ✅  | ⚪   |  ⚪  | ⚪     | ⚪     | ⚪      |

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
    cell -->|CCL| cluster;
    linkStyle 1 stroke: blue;

    %% CUDA CCL algorithm
    cell -->|CCL| cluster;
    linkStyle 2 stroke: green;

    %% CPU clusterization
    cluster -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/clusterization/measurement_creation.hpp'>Agg.</a>| meas;
    linkStyle 3 stroke: black;

    %% SYCL clusterization
    cluster -->|Agg.| meas;
    linkStyle 4 stroke: blue;

    %% CUDA clusterization
    cluster -->|Agg.| meas;
    linkStyle 5 stroke: green;

    %% CUDA CCA
    cell -->|CCA| meas;
    linkStyle 6 stroke: green;

    %% CPU local to global
    meas -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/clusterization/spacepoint_formation.hpp'>L2G</a>| sp;
    linkStyle 7 stroke: black;

    %% SYCL local to global
    meas -->|L2G| sp;
    linkStyle 8 stroke: blue;

    %% CUDA local to global
    meas -->|L2G| sp;
    linkStyle 9 stroke: green;

    %% CPU binning
    sp -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/seeding/spacepoint_binning.hpp'>Binning</a>| bin;
    linkStyle 10 stroke: black;

    %% CUDA binning
    sp -->|<a href='https://github.com/acts-project/traccc/blob/main/device/cuda/include/traccc/cuda/seeding/spacepoint_binning.hpp'>Binning</a>| bin;
    linkStyle 11 stroke: green;

    %% CPU seeding
    bin -->|Seeding| seed;
    linkStyle 12 stroke: black;

    %% SYCL seeding
    bin -->|<a href='https://github.com/acts-project/traccc/blob/main/device/sycl/include/traccc/sycl/seeding/seed_finding.hpp'>Seeding</a>| seed;
    linkStyle 13 stroke: blue;

    %% CUDA seeding
    bin -->|<a href='https://github.com/acts-project/traccc/tree/main/device/cuda/include/traccc/cuda/seeding'>Seeding</a>| seed;
    linkStyle 14 stroke: green;

    %% CUDA binless seeding
    sp -.->|Seeding| seed;
    linkStyle 15 stroke: green;

    %% CPU param est.
    seed -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/seeding/track_params_estimation.hpp'>Param. Est.</a>| ptrack;
    linkStyle 16 stroke: black;

    %% CUDA param est.
    seed -->|<a href='https://github.com/acts-project/traccc/blob/main/device/cuda/include/traccc/cuda/seeding/track_params_estimation.hpp'>Param. Est.</a>| ptrack;
    linkStyle 17 stroke: green;

    %% CPU CKF
    ptrack -.->|CKF| track;
    linkStyle 18 stroke: black;

    %% CPU Kalman filter
    track -->|<a href='https://github.com/acts-project/traccc/blob/main/core/include/traccc/fitting/fitting_algorithm.hpp'>Kalman filter</a>| track;
    linkStyle 19 stroke: black;

    %% CUDA Kalman filter
    track -->|<a href='https://github.com/acts-project/traccc/blob/main/device/cuda/include/traccc/cuda/fitting/fitting_algorithm.hpp'>Kalman filter</a>| track;
    linkStyle 20 stroke: green;

    %% SYCL binning
    sp -->|<a href='https://github.com/acts-project/traccc/blob/main/device/sycl/include/traccc/sycl/seeding/spacepoint_binning.hpp'>Binning</a>| bin;
    linkStyle 21 stroke: blue;

    %% SYCL track parameter est.
    seed -->|<a href='https://github.com/acts-project/traccc/blob/main/device/sycl/include/traccc/sycl/seeding/track_params_estimation.hpp'>Param. Est.</a>| ptrack;
    linkStyle 22 stroke: blue;

    %% Futhark measurement creation
    cell -->|<a href='https://github.com/acts-project/traccc/blob/main/device/futhark/src/measurement_creation.fut'>CCA</a>| meas;
    linkStyle 23 stroke: brown;

    %% Futhark spacepoint creation
    meas -->|<a href='https://github.com/acts-project/traccc/blob/main/device/futhark/src/spacepoint_formation.fut'>L2G</a>| sp;
    linkStyle 24 stroke: brown;

    %% SYCL Kalman filter
    track -->|<a href='https://github.com/acts-project/traccc/blob/main/device/sycl/include/traccc/sycl/fitting/fitting_algorithm.hpp'>Kalman filter</a>| track;
    linkStyle 25 stroke: blue;

    %% CUDA CKF
    ptrack -.->|CKF| track;
    linkStyle 26 stroke: green;
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

| Build | OS | gcc | CUDA | comment |
| --- | --- | --- | --- | --- |
| CUDA | Ubuntu 20.04   | 9.3.0 | 11.5 | runs on CI |

### Dependencies

- [Boost](https://www.boost.org/): program_options
- [CMake](https://cmake.org/)
- (Optional) [ROOT](https://root.cern/): RIO, Hist, Tree

## Getting started

### Clone the repository

Clone the repository and setup the data directory.

```sh
git clone git@github.com:acts-project/traccc.git
cd traccc
./data/traccc_data_get_files.sh
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
| TRACCC_USE_ROOT | Build physics performance analysis code using an existing installation of ROOT from the build environment |

## Examples

### CPU reconstruction chain

```sh
<build_directory>/bin/traccc_seq_example --detector-file=tml_detector/trackml-detector.csv --digitization-config-file=tml_detector/default-geometric-config-generic.json --input-directory=tml_pixels/ --input-events=10

<build_directory>/bin/traccc_throughput_mt --detector-file=tml_detector/trackml-detector.csv --digitization-config-file=tml_detector/default-geometric-config-generic.json --input-directory=tml_pixels/  --cold-run-events=100 --processed-events=1000 --threads=1
```

### CUDA reconstruction chain

- Users can generate CUDA examples by adding `-DTRACCC_BUILD_CUDA=ON` to cmake options

```sh
<build_directory>/bin/traccc_seq_example_cuda --detector-file=tml_detector/trackml-detector.csv --digitization-config-file=tml_detector/default-geometric-config-generic.json --input-directory=tml_pixels/ --input--events=10 --run-cpu=1

<build_directory>/bin/traccc_throughput_mt_cuda --detector-file=tml_detector/trackml-detector.csv --digitization-config-file=tml_detector/default-geometric-config-generic.json --input-directory=tml_pixels/  --cold-run-events=100 --processed-events=1000 --threads=1
```

### SYCL reconstruction chain

- Users can generate SYCL examples by adding `-DTRACCC_BUILD_SYCL=ON` to cmake options

```sh
<build_directory>/bin/traccc_seq_example_sycl --detector-file=tml_detector/trackml-detector.csv --digitization-config-file=tml_detector/default-geometric-config-generic.json --input-directory=tml_pixels/ --input--events=10 --run-cpu=1

<build_directory>/bin/traccc_throughput_mt_sycl --detector-file=tml_detector/trackml-detector.csv --digitization-config-file=tml_detector/default-geometric-config-generic.json --input-directory=tml_pixels/  --cold-run-events=100 --processed-events=1000 --threads=1
```

### Running a partial chain with simplified simulation data

Users can generate muon-like particle simulation data with the pre-built detray geometries:

```sh
# Generate telescope geometry data
<build_directory>/bin/traccc_simulate_telescope --gen-vertex-xyz-mm=0:0:0 --gen-vertex-xyz-std-mm=0:0:0 --gen-mom-gev=100:100 --gen-phi-degree=0:0 --gen-events=10 --gen-nparticles=2000 --output-directory=detray_simulation/telescope_detector/n_particles_2000/ --gen-eta=1:3

# Generate toy geometry data
<build_directory>/bin/traccc_simulate_toy_detector --gen-vertex-xyz-mm=0:0:0 --gen-vertex-xyz-std-mm=0:0:0 --gen-mom-gev=100:100 --gen-phi-degree=0:360 --gen-events=10 --gen-nparticles=2000 --output-directory=detray_simulation/toy_detector/n_particles_2000/ --gen-eta=-3:3 --constraint-step-size-mm=1 --search-window 3:3

# Generate drift chamber data
<build_directory>/bin/traccc_simulate_wire_chamber --gen-vertex-xyz-mm=0:0:0 --gen-vertex-xyz-std-mm=0:0:0 --gen-mom-gev=2:2 --gen-phi-degree=0:360 --gen-events=10 --gen-nparticles=100 --output-directory=detray_simulation/wire_chamber/n_particles_100/ --gen-eta=-1:1 --constraint-step-size-mm=1 --search-window 3:3
```

The simulation will also generate the detector json files (geometry, material and surface_grid) in the current directory. It is user's responsibility to move them to an appropriate place (e.g. `<detector_directory>`) and match them to the input file arguments of reconstruction chains.

If users have a geometry json file already, it is also possible to run simulation with `traccc_simulate` application

```sh
# Given that users have a geometry json file
<build_directory>/bin/traccc_simulate  --output-directory=<output-directory>  --detector-file=<geometry_file> --material-file=<material-file> --grid-file=<grid-file>  --event=10 --constraint-step-size-mm=1
```

There are three types of partial reconstruction chain users can operate: `seeding_example`, `truth_finding_example`, and `truth_fitting_example` where their algorithm coverages are shown in the table below. Each of them starts from truth measurements, truth seeds, and truth tracks, respectively.

| Category                | Clusterization | Seeding | Track finding | Track fitting |
| ----------------------- | -------------- | ------- | ------------- | ------------- |
| `seeding_example`       |                | ✅      | ✅            | ✅            |
| `truth_finding_example` |                |         | ✅            | ✅            |
| `truth_fitting_example` |                |         |               | ✅            |

The dirft chamber will not produce meaningful results with `seeding_example` as the current seeding algorithm is only designed for 2D measurement objects. Truth finding works OK in general but the combinatoric explosion can occur for a few unlucky events, leading to poor pull value distributions. The followings are example commands:

```sh
# Run cuda seeding example for toy geometry
<build_directory>/bin/traccc_seeding_example_cuda --input-directory=detray_simulation/toy_detector/n_particles_2000/ --check-performance --detector-file=<detector_directory>/toy_detector_geometry.json --material-file=<detector_directory>/toy_detector_homogeneous_material.json --grid-file=<detector_directory>/toy_detector_surface_grids.json --input-events=1 --track-candidates-range=3:30 --constraint-step-size-mm=1000 --run-cpu=1 --search-window 3:3
```

```sh
# Run cuda truth finding example for toy geometry
<build_directory>/bin/traccc_truth_finding_example_cuda --input-directory=detray_simulation/toy_detector/n_particles_2000/ --check-performance --detector-file=<detector_directory>/toy_detector_geometry.json --material-file=<detector_directory>/toy_detector_homogeneous_material.json --grid-file=<detector_directory>/toy_detector_surface_grids.json --input-events=1 --track-candidates-range=3:30 --constraint-step-size-mm=1000 --run-cpu=1 --search-window 3:3
```

```sh
# Run cuda truth finding example for drift chamber
<build_directory>/bin/traccc_truth_finding_example_cuda --input-directory=detray_simulation/wire_chamber/n_particles_100/ --check-performance --detector-file=<detector_directory>/wire_chamber_geometry.json --material-file=<detector_directory>/wire_chamber_homogeneous_material.json --grid-file=<detector_directory>/wire_chamber_surface_grids.json  --input-events=10 --track-candidates-range=6:30 --constraint-step-size-mm=1 --run-cpu=1 --search-window 3:3
```

```sh
# Run cpu truth fitting example for drift chamber
<build_directory>/bin/traccc_truth_fitting_example --input-directory=detray_simulation/wire_chamber/n_particles_2000_100GeV/ --check-performance --detector-file=<detector_directory>/wire_chamber_geometry.json --material-file=<detector_directory>/wire_chamber_homogeneous_material.json --grid-file=<detector_directory>/wire_chamber_surface_grids.json --input-events=10 --constraint-step-size-mm=1 --search-window 3:3
```

Users can open the performance root files (with `--check-performance=true`) and draw the histograms.

```sh
$ root -l performance_track_finding.root
root [0]
Attaching file performance_track_finding.root as _file0...
(TFile *) 0x3871910
root [1] finding_trackeff_vs_eta->Draw()
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
