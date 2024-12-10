/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

/// @name Preprocessor macros with version information
/// @{

/// Major version number of the VecMem project
#define VECMEM_VERSION_MAJOR 1
/// Minor version number of the VecMem project
#define VECMEM_VERSION_MINOR 13
/// Patch version number of the VecMem project
#define VECMEM_VERSION_PATCH 0
/// Version number of the VecMem project
#define VECMEM_VERSION "1.13.0"

/// @}

namespace vecmem {

/// @name C++ variables with version information
/// @{

/// Major version number of the VecMem project
static constexpr int version_major = 1;
/// Minor version number of the VecMem project
static constexpr int version_minor = 13;
/// Patch version number of the VecMem project
static constexpr int version_patch = 0;
/// Version number of the VecMem project
static constexpr char version[] = "1.13.0";

/// @}

}  // namespace vecmem
