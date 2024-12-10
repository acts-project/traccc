# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

find_package(Boost 1.71.0 REQUIRED COMPONENTS filesystem)

add_library(testing_utils tmp_file.cpp)

# Ensure that the utils.
target_link_libraries(testing_utils PUBLIC Boost::filesystem)

target_include_directories(
    testing_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_definitions(testing_utils PRIVATE _CRT_SECURE_NO_WARNINGS)
