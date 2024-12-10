# Algebra plugins library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( CPack )

# Export the configuration of the project.
include( CMakePackageConfigHelpers )
set( CMAKE_INSTALL_CMAKEDIR
   "${CMAKE_INSTALL_LIBDIR}/cmake/algebra-plugins-${PROJECT_VERSION}" )
install( EXPORT algebra-plugins-exports
   NAMESPACE "algebra::"
   FILE "algebra-plugins-config-targets.cmake"
   DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )
configure_package_config_file(
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/algebra-plugins-config.cmake.in"
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/algebra-plugins-config.cmake"
   INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
   PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
             CMAKE_INSTALL_CMAKEDIR
   NO_CHECK_REQUIRED_COMPONENTS_MACRO )
write_basic_package_version_file(
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/algebra-plugins-config-version.cmake"
   COMPATIBILITY "AnyNewerVersion" )
install( FILES
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/algebra-plugins-config.cmake"
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/algebra-plugins-config-version.cmake"
   DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )
