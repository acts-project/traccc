# VecMem project, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
include( CPack )

# Export the configuration of the project.
include( CMakePackageConfigHelpers )
install( EXPORT vecmem-exports
   NAMESPACE "vecmem::"
   FILE "vecmem-config-targets.cmake"
   DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )
configure_package_config_file(
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vecmem-config.cmake.in"
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecmem-config.cmake"
   INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
   PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
             CMAKE_INSTALL_CMAKEDIR
   NO_CHECK_REQUIRED_COMPONENTS_MACRO )
write_basic_package_version_file(
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecmem-config-version.cmake"
   COMPATIBILITY "AnyNewerVersion" )
install( FILES
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecmem-config.cmake"
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vecmem-config-version.cmake"
   DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )

# Install the "language helper" files.
install( FILES
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vecmem-check-language.cmake"
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vecmem-check-sycl-code-compiles.cmake"
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vecmem-check-sycl-source-compiles.cmake"
   DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip"
                   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sycl"
   DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )

# Clean up.
unset( CMAKE_INSTALL_CMAKEDIR )
