# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.18)

project("covfie_downstream" VERSION 0.0.0)

find_package(covfie REQUIRED)

add_executable(main main.cpp)

target_link_libraries(
    main
    covfie::core
    covfie::cpu
)

if(TARGET core)
    message(FATAL_ERROR "Target `core` should not exist.")
endif()

if(TARGET cpu)
    message(FATAL_ERROR "Target `cpu` should not exist.")
endif()

if(TARGET benchmark_cpu)
    message(FATAL_ERROR "Target `benchmark_cpu` should not exist.")
endif()

if(TARGET test_core)
    message(FATAL_ERROR "Target `test_core` should not exist.")
endif()
