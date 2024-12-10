# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

add_subdirectory(core)

if(COVFIE_PLATFORM_CPU)
    add_subdirectory(cpu)
endif()

if(COVFIE_PLATFORM_CUDA)
    add_subdirectory(cuda)
endif()
