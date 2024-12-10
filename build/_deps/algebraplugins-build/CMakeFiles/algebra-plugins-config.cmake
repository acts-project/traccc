# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the helper functions/macros.

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was algebra-plugins-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################

# Set up variables describing which plugins were enabled during the Algebra
# Plugins build.
set( ALGEBRA_PLUGINS_INCLUDE_EIGEN TRUE )
set( ALGEBRA_PLUGINS_INCLUDE_SMATRIX OFF )
set( ALGEBRA_PLUGINS_INCLUDE_VC OFF )
set( ALGEBRA_PLUGINS_INCLUDE_VECMEM TRUE )

# Set up some simple variables for using the package.
set( algebra_plugins_VERSION "0.25.1" )
set_and_check( algebra_plugins_INCLUDE_DIR
   "${PACKAGE_PREFIX_DIR}/include" )
set_and_check( algebra_plugins_CMAKE_DIR "${PACKAGE_PREFIX_DIR}/lib/cmake/algebra-plugins-0.25.1" )

# Find all packages that Algebra Plugins needs to function.
include( CMakeFindDependencyMacro )
if( ALGEBRA_PLUGINS_INCLUDE_EIGEN )
   find_dependency( Eigen3 )
endif()
if( ALGEBRA_PLUGINS_INCLUDE_SMATRIX )
   find_dependency( ROOT COMPONENTS Smatrix )
endif()
if( ALGEBRA_PLUGINS_INCLUDE_VC )
   find_dependency( Vc 1.4.2 )
endif()
if( ALGEBRA_PLUGINS_INCLUDE_VECMEM )
   find_dependency( vecmem )
endif()

# Print a standard information message about the package being found.
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( algebra-plugins REQUIRED_VARS
   CMAKE_CURRENT_LIST_FILE
   VERSION_VAR algebra_plugins_VERSION )

# Include the file listing all the imported targets and options.
include( "${algebra_plugins_CMAKE_DIR}/algebra-plugins-config-targets.cmake" )
