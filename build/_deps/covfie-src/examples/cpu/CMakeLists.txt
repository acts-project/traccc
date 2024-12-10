# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# Add the 3D field slice rendering tool.
add_executable(render_slice_cpu render_slice.cpp)

target_link_libraries(
    render_slice_cpu
    PRIVATE
        covfie_core
        covfie_cpu
        bitmap
        Boost::log
        Boost::log_setup
        Boost::program_options
)

add_executable(render_image_cpu render_image.cpp)

target_link_libraries(
    render_image_cpu
    PRIVATE
        covfie_core
        covfie_cpu
        bitmap
        Boost::log
        Boost::log_setup
        Boost::program_options
)
