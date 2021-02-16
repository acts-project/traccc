# traccc

Demonstrator tracking chain for accelerators.

## Getting started

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

