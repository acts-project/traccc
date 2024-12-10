# This file is part of the ACTS project.
#
# Copyright (C) 2016 CERN for the benefit of the ACTS project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This script sets up the ACTS Examples environment in a somewhat robust way.

if [ -n "$ZSH_VERSION" ]; then
    script_dir=${0:a:h}
elif [ -n "$BASH_VERSION" ]; then
    script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
else
    # If the current shell is not ZSH or Bash, we can't guarantee that the
    # script will work, so we throw an error.
    echo "ERROR:   neither ZSH nor Bash was detected, other shells are not supported. The environment has not been modified."
    exit 1
fi

# source the python and ODD environment
# first check if python bindings are enabled
# then check if the setup.sh file exists
if [ "OFF" = "ON" ]; then
    if [[ -f "$script_dir/python/setup.sh" ]]; then
      . "$script_dir/python/setup.sh"
    else
      echo "ERROR:   setup.sh for python and ODD is missing."
      exit 1
    fi
else
  echo "INFO:    Python bindings are disabled."
fi

# set ACTS and ODD environment variables
export ACTS_SOURCE_DIR=/home/slobod/Documents/Work/traccc/build/_deps/acts-src
export ODD_SOURCE_DIR=/home/slobod/Documents/Work/traccc/build/_deps/acts-src/thirdparty/OpenDataDetector

# make ACTS binaries available
export PATH="$script_dir/bin:${PATH}"
export LD_LIBRARY_PATH="$script_dir/lib:${LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="$script_dir/lib:${DYLD_LIBRARY_PATH}"

# activate dependencies if present
if [[ -d "ROOT_DIR-NOTFOUND" ]]; then
  . /thisroot.sh
fi
if [[ -d "" ]]; then
  . /../../bin/geant4.sh
fi
if [[ -d "" ]]; then
  . /../bin/thisdd4hep.sh
fi
if [[ -d "" ]]; then
  export LD_LIBRARY_PATH=":${LD_LIBRARY_PATH}"
  export DYLD_LIBRARY_PATH=":${DYLD_LIBRARY_PATH}"
  export ROOT_INCLUDE_PATH=":${ROOT_INCLUDE_PATH}"
  export PYTHONPATH=":${PYTHONPATH}"
fi
if [[ -d "" ]]; then
  export LD_LIBRARY_PATH="/lib:${LD_LIBRARY_PATH}"
  export DYLD_LIBRARY_PATH="/lib:${DYLD_LIBRARY_PATH}"
  export ROOT_INCLUDE_PATH=":${ROOT_INCLUDE_PATH}"
fi
