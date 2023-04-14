# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022-2023 CERN for the benefit of the ACTS project
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

# Make sure that GNU Make and CTest would use all available cores.
export MAKEFLAGS="-j`nproc`"
export CTEST_PARALLEL_LEVEL=`nproc`
