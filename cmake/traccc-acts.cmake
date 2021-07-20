# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard(GLOBAL)

# Try to find ACTS installed on the system.
find_package(Acts QUIET)

# If it was found, then we're finished.
if(Acts_FOUND)
   # Call find_package again, just to nicely print where it is picked up from.
   find_package(Acts)
   return()
endif()

# Tell the user what's happening.
message(STATUS "Building ACTS as part of the traccc project")

# Declare where to get ACTS from.
FetchContent_Declare(
   Acts
   GIT_REPOSITORY "https://github.com/acts-project/acts.git"
   GIT_TAG "v9.2.0"
)

# Make the ACTS source available to the rest of the build system.
FetchContent_MakeAvailable(Acts)
