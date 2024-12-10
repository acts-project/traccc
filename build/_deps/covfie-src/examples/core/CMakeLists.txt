# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# Add the conversion of B-fields to our native format.
add_executable(convert_bfield convert_bfield.cpp)

target_link_libraries(
    convert_bfield
    PRIVATE
        covfie_core
        Boost::log
        Boost::log_setup
        Boost::program_options
)

# Add the conversion of B-fields from CSV to our native format.
add_executable(convert_bfield_csv convert_bfield_csv.cpp)

target_link_libraries(
    convert_bfield_csv
    PRIVATE
        covfie_core
        Boost::log
        Boost::log_setup
        Boost::program_options
)

# Add the first example that can be found in the README.
add_executable(readme_example_1 readme_example_1.cpp)

target_link_libraries(readme_example_1 PRIVATE covfie_core)

# Add the second example that can be found in the README.
add_executable(readme_example_2 readme_example_2.cpp)

target_link_libraries(readme_example_2 PRIVATE covfie_core)

# Add the an executable to create 2D slices from 3D fields.
add_executable(slice3dto2d slice3dto2d.cpp)

target_link_libraries(
    slice3dto2d
    PRIVATE
        covfie_core
        Boost::log
        Boost::log_setup
        Boost::program_options
)

# Add the an executable to scale up B-fields.
add_executable(scaleup_bfield scaleup_bfield.cpp)

target_link_libraries(
    scaleup_bfield
    PRIVATE
        covfie_core
        Boost::log
        Boost::log_setup
        Boost::program_options
)

add_library(asm asm.cpp)
target_link_libraries(asm PRIVATE covfie_core)

# Some libraries which are designed to inspected by the user.
add_library(shuffle_asm shuffle_asm.cpp)
target_link_libraries(shuffle_asm PRIVATE covfie_core)

# Executable for generating an testable field.
add_executable(generate_test_field generate_test_field.cpp)

target_link_libraries(
    generate_test_field
    PRIVATE
        covfie_core
        Boost::log
        Boost::log_setup
        Boost::program_options
)
