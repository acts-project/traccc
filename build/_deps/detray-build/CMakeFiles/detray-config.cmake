# Detray library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the helper functions/macros.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was detray-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################

# Set up variables describing which components were enabled during the
# Detray build.
set( DETRAY_EIGEN_PLUGIN TRUE )
set( DETRAY_SMATRIX_PLUGIN OFF )
set( DETRAY_VC_AOS_PLUGIN OFF )
set( DETRAY_VC_SOA_PLUGIN OFF )
set( DETRAY_DISPLAY  )

# Set up some simple variables for using the package.
set( detray_VERSION "0.83.0" )
set_and_check( detray_INCLUDE_DIR
   "${PACKAGE_PREFIX_DIR}/include" )
set_and_check( detray_CMAKE_DIR "${PACKAGE_PREFIX_DIR}/lib/cmake/detray-0.83.0" )

# Find all packages that Detray needs to function.
include( CMakeFindDependencyMacro )
find_dependency( algebra-plugins )
find_dependency( covfie )
find_dependency( vecmem )
find_dependency( dfelibs )
find_dependency( nlohmann_json )
if( DETRAY_DISPLAY )
   find_dependency( actsvg )
endif()

# Print a standard information message about the package being found.
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( detray REQUIRED_VARS
   CMAKE_CURRENT_LIST_FILE
   VERSION_VAR detray_VERSION )

# Include the file listing all the imported targets and options.
include( "${detray_CMAKE_DIR}/detray-config-targets.cmake" )
