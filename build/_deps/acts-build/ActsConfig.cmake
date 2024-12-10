# CMake config for the Acts package
#
# Defines CMake targets for requested and available components.  All additional
# information, e.g. include directories and dependencies, are defined as
# target-specific properties and are automatically propagated when linking to
# the target.
#
# Defines the following additional variables:
#
#   - Acts_COMPONENTS
#   - Acts_COMMIT_HASH
#   - Acts_COMMIT_HASH_SHORT


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ActsConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(Acts_COMPONENTS Core;PluginJson)
set(Acts_COMMIT_HASH "c9d68283f4f33a8ab4021cc656ee35b8ee77827c")
set(Acts_COMMIT_HASH_SHORT "c9d68283f")

# print version and components information
if(NOT Acts_FIND_QUIETLY)
  message(STATUS "found Acts version ${Acts_VERSION} commit ${Acts_COMMIT_HASH_SHORT}")
endif()

# check that requested components are available
foreach(_component ${Acts_FIND_COMPONENTS})
  # check if this component is available
  if(NOT _component IN_LIST Acts_COMPONENTS)
    if (${Acts_FIND_REQUIRED_${_component}})
      # not supported, but required -> fail
      set(Acts_FOUND False)
      set(Acts_NOT_FOUND_MESSAGE "required component '${_component}' not found")
    else()
      # not supported and optional -> skip
      list(REMOVE_ITEM Acts_FIND_COMPONENTS ${_component})
      if(NOT Acts_FIND_QUIETLY)
        message(STATUS "optional component '${_component}' not found")
      endif()
    endif()
  endif()
endforeach()

# add this to the current CMAKE_MODULE_PATH to find third party modules
# that not provide a XXXConfig.cmake or XXX-config.cmake file
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/Modules)

# find external dependencies that are needed to link with Acts. since the
# exported Acts targets only store the linked external target names they need
# to be found again. this avoids hard-coded paths and makes the installed
# config/library relocatable. use exact version match where possible to ensure
# the same versions are found that were used at build time.
# `find_dependency` is a wrapper around `find_package` that automatically
# handles QUIET and REQUIRED parameters.
include(CMakeFindDependencyMacro)
find_dependency(Boost 1.86.0 CONFIG EXACT)
if(ON)
  find_dependency(Eigen3  CONFIG EXACT)
endif()
if(PluginDD4hep IN_LIST Acts_COMPONENTS)
  find_dependency(DD4hep  CONFIG EXACT)
endif()
if(PluginJson IN_LIST Acts_COMPONENTS)
  find_dependency(nlohmann_json  CONFIG EXACT)
endif()
if(PluginTGeo IN_LIST Acts_COMPONENTS)
  find_dependency(ROOT  CONFIG EXACT)
endif()
if(PluginActSVG IN_LIST Acts_COMPONENTS)
  find_dependency(actsvg  CONFIG EXACT)
endif()
if(PluginEDM4hep IN_LIST Acts_COMPONENTS)
  find_dependency(EDM4HEP  CONFIG EXACT)
endif()
if(PluginPodio IN_LIST Acts_COMPONENTS)
  find_dependency(podio  CONFIG EXACT)
endif()
if(PluginGeoModel IN_LIST Acts_COMPONENTS)
  find_dependency(GeoModelCore  CONFIG EXACT)
  find_dependency(GeoModelIO  CONFIG EXACT)
endif()
if (PluginHashing IN_LIST Acts_COMPONENTS)
  find_dependency(Annoy  CONFIG EXACT)
endif()

# dependencies that we have built ourselves but cannot be
# straightforwardly handed to cmake
if(NOT ON)
  add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
  target_include_directories(Eigen3::Eigen INTERFACE "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
endif()

if(PluginPodio IN_LIST Acts_COMPONENTS)
  include(${CMAKE_CURRENT_LIST_DIR}/ActsPodioEdmTargets.cmake)
endif()

if(PluginDetray IN_LIST Acts_COMPONENTS)
  find_dependency(vecmem  CONFIG EXACT)
  find_dependency(covfie  CONFIG EXACT)
  find_dependency(algebra-plugins  CONFIG EXACT)
  find_dependency(actsvg  CONFIG EXACT)
  find_dependency(detray  CONFIG EXACT)
endif()

if (PluginCovfie IN_LIST Acts_COMPONENTS)
  find_dependency(covfie  CONFIG EXACT)
endif()

# load **all** available components. we can not just include the requested
# components since there can be interdependencies between them.
if(NOT Acts_FIND_QUIETLY)
  message(STATUS "loading components:")
endif()
foreach(_component ${Acts_COMPONENTS})
  if(NOT Acts_FIND_QUIETLY)
    message(STATUS "  ${_component}")
  endif()
  # include the targets file to create the imported targets for the user
  include(${CMAKE_CURRENT_LIST_DIR}/Acts${_component}Targets.cmake)
endforeach()
