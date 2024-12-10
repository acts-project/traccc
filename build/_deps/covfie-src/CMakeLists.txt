# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.21)

project("covfie" VERSION 0.10.0)

# Load some dependencies.
include(GNUInstallDirs)

# Declare options which control the parts of the code being built.
option(COVFIE_BUILD_BENCHMARKS "Build benchmark executables.")
option(COVFIE_BUILD_TESTS "Build test executables.")
option(COVFIE_BUILD_EXAMPLES "Build example executables.")

# Declare option to enable header completeness tests.
option(COVFIE_TEST_HEADERS "Enable header completeness tests.")

# Declare options for the different platforms that we wish to support.
option(COVFIE_PLATFORM_CPU "Enable building of CPU code." On)
option(COVFIE_PLATFORM_CUDA "Enable building of CUDA code.")

# Additional options that may be useful in some cases, such as CI.
option(
    COVFIE_QUIET
    "Disable warnings about missing C++ features. Enabling this is strongly discouraged."
)

option(COVFIE_FAIL_ON_WARNINGS "Treat compiler warnings as errors.")

# Make the CMake modules in the cmake/ directory visible to the project.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# The core library should always be built.
add_subdirectory(lib)

# The benchmarks should be build only if requested...
if(COVFIE_BUILD_BENCHMARKS)
    if(NOT PROJECT_IS_TOP_LEVEL)
        message(
            FATAL_ERROR
            "Build of benchmarks was requested, but covfie is not the top level project."
        )
    endif()

    add_subdirectory(benchmarks)
endif()

# ...the same goes for the tests...
if(COVFIE_BUILD_TESTS)
    if(NOT PROJECT_IS_TOP_LEVEL)
        message(
            FATAL_ERROR
            "Build of tests was requested, but covfie is not the top level project."
        )
    endif()

    add_subdirectory(tests)
endif()

# ...and the examples.
if(COVFIE_BUILD_EXAMPLES)
    if(NOT PROJECT_IS_TOP_LEVEL)
        message(
            FATAL_ERROR
            "Build of examples was requested, but covfie is not the top level project."
        )
    endif()

    add_subdirectory(examples)
endif()

# Installation logic.
# CMake is hell.
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)
