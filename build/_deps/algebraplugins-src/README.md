# algebra-plugins

This repository provides different algebra plugins with minimal functionality
for the R&D projects [detray](https://github.com/acts-project/detray) and
[traccc](https://github.com/acts-project/traccc).

| Backend                                                                   | CPU | CUDA | SYCL |
| ------------------------------------------------------------------------- | --- | ---- | ---- |
| cmath                                                                     | ✅  | ✅  | ✅   |
| [Eigen](https://eigen.tuxfamily.org)                                      | ✅  | ✅  | ✅   |
| [SMatrix](https://root.cern.ch/doc/master/group__SMatrixGroup.html)       | ✅  | ⚪  | ⚪   |
| [VC](https://github.com/VcDevel/Vc)                                       | ✅  | ⚪  | ⚪   |
| [Fastor](https://github.com/romeric/Fastor)                               | ✅  | ⚪  | ⚪   |

## Building

To build it standalone, run e.g.

```
git clone https://github.com/acts-project/algebra-plugins.git
cmake -DCMAKE_BUILD_TYPE=Release -S algebra-plugins -B algebra-plugins-build
cmake --build algebra-plugins-build
```

Available options:

- `ALGEBRA_PLUGINS_INCLUDE_<XXX>`: Boolean to turn on/off the build of one of
  the following plugins:
  * `EIGEN`: Plugin using [Eigen](https://eigen.tuxfamily.org)
    (`OFF` by default)
  * `SMATRIX`: Plugin using
    [SMatrix](https://root.cern/doc/master/group__SMatrixGroup.html)
    (`OFF` by default)
  * `VC`: Plugin using [Vc](https://github.com/VcDevel/Vc)
    (`OFF` by default)
  * `FASTOR`: Plugin using [Fastor](https://github.com/romeric/Fastor)
    (`OFF` by default)
  * `VECMEM`: Plugin using [VecMem](https://github.com/acts-project/vecmem)
    (`OFF` by default)
- `ALGEBRA_PLUGINS_USE_SYSTEM_LIBS`: Boolean configuring whether to search for all external libraries "on the system" or not
- `ALGEBRA_PLUGINS_SETUP_<XXX>`: Boolean to turn on/off the explicit "setup" of
  the external libraries (`GOOGLETEST`, [`BENCHMARK`](https://github.com/google/benchmark), `EIGEN3`, `VC`, `FASTOR`, and `VECMEM`)
- `ALGEBRA_PLUGINS_USE_SYSTEM_<XXX>`: Boolean configuring how to set up a given
  external library
  * `ON`: The external library is searched for "on the system" using
    [find_package](https://cmake.org/cmake/help/latest/command/find_package.html);
  * `OFF`: The package is set up for build as part of this project, using
    [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html).
- `ALGEBRA_PLUGINS_BUILD_TESTING`: Turn the build/setup of the unit tests on/off
  (`ON` by default)
