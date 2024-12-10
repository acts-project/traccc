# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

add_library(benchmark test_field.cpp)

target_include_directories(benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(benchmark PUBLIC covfie_core)

target_compile_definitions(benchmark PRIVATE _CRT_SECURE_NO_WARNINGS)
