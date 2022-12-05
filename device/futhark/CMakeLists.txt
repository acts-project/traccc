# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

find_package(Futhark REQUIRED)

# Create a new traccc library which is built from the CCA CUDA core, plus some
# extra code.
traccc_add_library(
    traccc_futhark
    futhark
    TYPE SHARED
    "src/context.cpp"
    "src/component_connection.cpp"
    "src/spacepoint_formation.cpp"
    "include/traccc/futhark/context.hpp"
    "include/traccc/futhark/component_connection.hpp"
    "include/traccc/futhark/spacepoint_formation.hpp"
    "include/traccc/futhark/utils.hpp"
    "include/traccc/futhark/wrapper.hpp"
)

# Link the new outward-facing library against our code librar(y|ies)
target_link_libraries(
    traccc_futhark
    PUBLIC
    traccc::core
    PRIVATE
    vecmem::core
)

# Add the Futhark sources to our traccc library.
add_futhark_to_library(
    traccc_futhark
    CUDA
    src/entry.fut
    DEPENDENCIES
    src/linear.fut
    src/edm.fut
    src/maybe.fut
    src/zip.fut
    src/measurement_creation.fut
    src/spacepoint_formation.fut
)
