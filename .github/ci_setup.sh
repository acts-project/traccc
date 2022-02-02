# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0
#
# This script is meant to configure the build/runtime environment of the
# Docker contaners that are used in the project's CI configuration.
#
# Usage: source .github/ci_setup.sh <platform name>
#

# The platform name.
PLATFORM_NAME=$1

# Set up the correct environment for the SYCL tests.
if [ "${PLATFORM_NAME}" = "SYCL" ]; then
   source /opt/intel/oneapi/setvars.sh
   export CXX=`which clang++`
   export SYCLCXX=`which dpcpp`
   export SYCL_DEVICE_FILTER=host
   # This is a hack to make Acts's "--coverage" flag work correctly with
   # oneAPI 2021.2.0. It may be removed once we update to a version of oneAPI
   # in the CI that does not have this issue.
   export LDFLAGS="-L${CMPLR_ROOT}/linux/compiler/lib/intel64_lin -lirc"
fi

# Make sure that GNU Make and CTest would use all available cores.
export MAKEFLAGS="-j`nproc`"
export CTEST_PARALLEL_LEVEL=`nproc`
