# Detray library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

message(STATUS "Building 'detray::test_utils' component")

# Set the common C++ flags.
include(detray-compiler-options-cpp)

include_directories(
    SYSTEM
    $<TARGET_PROPERTY:covfie::core,INTERFACE_INCLUDE_DIRECTORIES>
)
include_directories(
    SYSTEM
    $<TARGET_PROPERTY:dfelibs::dfelibs,INTERFACE_INCLUDE_DIRECTORIES>
)

# Set up a test library, which can be used by downstream projects without
# building the detray unit and integration
if(DETRAY_BUILD_TEST_UTILS)
    file(
        GLOB _detray_test_utils_headers
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "include/detray/test/utils/*.hpp"
        "include/detray/test/utils/detectors/*.hpp"
        "include/detray/test/utils/detectors/factories/*.hpp"
        "include/detray/test/utils/simulation/event_generator/*.hpp"
        "include/detray/test/utils/simulation/*.hpp"
    )

    detray_add_library(detray_test_utils test_utils ${_detray_test_utils_headers})
    target_link_libraries(
        detray_test_utils
        INTERFACE vecmem::core detray::core detray::detectors
    )

    unset(_detray_test_utils_headers)
endif()

# Build extended validation tools
if(DETRAY_BUILD_VALIDATION_TOOLS)
    file(
        GLOB _detray_validation_headers
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "include/detray/test/validation/*.hpp"
    )

    detray_add_library(detray_validation_utils validation_utils ${_detray_validation_headers})
    target_link_libraries(
        detray_validation_utils
        INTERFACE
            vecmem::core
            detray::core
            detray::test_utils
            detray::csv_io
            detray::svgtools
    )

    unset(_detray_validation_headers)
endif()

# Build the common (host and device) test library (mostly for integration tests)
if(DETRAY_BUILD_TESTING)
    message(STATUS "Building 'detray::test_common' component")

    file(
        GLOB _detray_test_headers
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "include/detray/test/common/*.hpp"
        "include/detray/test/common/detail/*.hpp"
    )

    add_library(detray_test_common INTERFACE "${_detray_test_headers}")
    add_library(detray::test_common ALIAS detray_test_common)

    target_include_directories(
        detray_test_common
        INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

    target_link_libraries(
        detray_test_common
        INTERFACE GTest::gtest vecmem::core detray::core detray::test_utils
    )

    unset(_detray_test_headers)
endif()

# Include all of the code-holding sub-directories.

# Build the benchmarks
if(DETRAY_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Build tool executables
if(DETRAY_BUILD_TESTING)
    # Build host and device specific test code
    add_subdirectory(include/detray/test/cpu)
    add_subdirectory(include/detray/test/device)

    # Build the unittests
    if(DETRAY_BUILD_UNITTESTS)
        add_subdirectory(unit_tests)
    endif()

    # Buid the integration tests
    if(DETRAY_BUILD_INTEGRATIONTESTS)
        add_subdirectory(integration_tests)
    endif()

    # Build the command line tools (also required for the host jacobian
    # validation integration test)
    if(
        (DETRAY_BUILD_INTEGRATIONTESTS AND DETRAY_BUILD_HOST)
        OR DETRAY_BUILD_CLI_TOOLS
    )
        add_subdirectory(tools)
    endif()
endif()
