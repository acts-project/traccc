# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

add_library(covfie_cpu INTERFACE)

target_include_directories(
    covfie_cpu
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(covfie_cpu INTERFACE cxx_std_20)

# Logic to ensure that the CPU module can be installed properly.
set_target_properties(
    covfie_cpu
    PROPERTIES
        EXPORT_NAME
            cpu
)

install(TARGETS covfie_cpu EXPORT ${PROJECT_NAME}Targets)

install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/covfie
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

target_link_libraries(covfie_cpu INTERFACE covfie_core)

# Hack for compatibility
if(NOT PROJECT_IS_TOP_LEVEL)
    add_library(covfie::cpu ALIAS covfie_cpu)
endif()

# Test the public headers of covfie::cpu.
if(COVFIE_TEST_HEADERS)
    include(covfie-functions)

    file(
        GLOB_RECURSE public_headers
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/covfie/*.hpp"
    )

    covfie_test_public_headers(
        covfie_cpu
        "${public_headers}"
    )
endif()
