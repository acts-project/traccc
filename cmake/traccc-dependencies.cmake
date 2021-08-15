# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

include_guard(GLOBAL)

# CMake include(s).
cmake_minimum_required(VERSION 3.11)
include(FetchContent)

# Include the CMake scripts for vecmem and the ACTS core.
include(traccc-vecmem)
include(traccc-acts)
