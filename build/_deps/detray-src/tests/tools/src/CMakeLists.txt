# Detray library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

if(DETRAY_BUILD_HOST)
    add_subdirectory(cpu)
endif()

if(DETRAY_BUILD_CUDA)
    add_subdirectory(cuda)
endif()
