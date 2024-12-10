# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
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
   source /opt/intel/oneapi/setvars.sh --include-intel-llvm
   # Use clang/clang++ instead of icx/icpx, to avoid some aggressive math
   # optimizations that break some tests.
   export CC=`which clang`
   export CXX=`which clang++`
   export SYCLCXX="${CXX} -fsycl"
   export SYCL_DEVICE_FILTER=host
fi
