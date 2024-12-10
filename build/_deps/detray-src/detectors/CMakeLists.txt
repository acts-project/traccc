# Detray library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

message(STATUS "Building 'detray::detectors' component")

# Set the common C++ flags.
include(detray-compiler-options-cpp)

# Set up a detector library, which contains the different detector types that
# are currently available by default.
file(
    GLOB _detray_detector_headers
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "include/detray/detectors/*.hpp"
)

detray_add_library( detray_detectors detectors ${_detray_detectors_headers})
target_link_libraries(
    detray_detectors
    INTERFACE covfie::core vecmem::core detray::core detray::io
)

unset(_detray_detector_headers)
