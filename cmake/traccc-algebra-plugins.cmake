# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Guard against multiple includes.
include_guard( GLOBAL )

# Algebra plugins
set(ALGEBRA_PLUGIN_INCLUDE_ARRAY ON CACHE INTERNAL "")
set(ALGEBRA_PLUGIN_INCLUDE_EIGEN ON CACHE INTERNAL "")
set(ALGEBRA_PLUGIN_INCLUDE_SMATRIX OFF CACHE INTERNAL "")
set(ALGEBRA_PLUGIN_INCLUDE_VC OFF CACHE INTERNAL "")
