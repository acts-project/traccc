# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Turn off the installation of GoogleTest.
option(INSTALL_GTEST "Flag controlling the installation of GTest" OFF)

# Needed once the benachmark external will be added to the build.
#set(BENCHMARK_ENABLE_INSTALL OFF)
#set(BENCHMARK_ENABLE_TESTING OFF)

# Add the external submodules.
if (NOT TARGET dfelibs)
  add_subdirectory(dfelibs)
endif()
if (NOT TARGET gtest)
  add_subdirectory(googletest)
endif()

# enable vecmem for algebra plugins
if (NOT ${ALGEBRA_PLUGIN_SOURCE_DIR})
  SET(ALGEBRA_PLUGIN_USE_VECMEM ON CACHE BOOL "Use Vecmem for array algebra")
  add_subdirectory(algebra-plugins)
endif()  
