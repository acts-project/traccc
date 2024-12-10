# Detray library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Temporary setting for the detray::scalar type, until it can be removed.
set(DETRAY_CUSTOM_SCALARTYPE
    "double"
    CACHE STRING
    "Scalar type to use in the Detray code"
)

# Add all subdirectories.
add_subdirectory(array)
if(DETRAY_EIGEN_PLUGIN)
    add_subdirectory(eigen)
endif()
if(DETRAY_SMATRIX_PLUGIN)
    add_subdirectory(smatrix)
endif()
if(DETRAY_VC_AOS_PLUGIN)
    add_subdirectory(vc_aos)
endif()
if(DETRAY_VC_SOA_PLUGIN)
    add_subdirectory(vc_soa)
endif()
