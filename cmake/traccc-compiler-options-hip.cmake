# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Include the helper function(s).
include( traccc-functions )

#HIP requires position-independent executables for linking
traccc_add_flag( CMAKE_HIP_FLAGS "-fPIE" )

