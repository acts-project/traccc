/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/types.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <cstddef>

namespace vecmem {
namespace details {

/// Helper function checking if a given pointer has a given alignment
///
/// It is used for debugging/validation by some parts of the code, it is
/// not meant to be used by clients of the library directly.
///
/// @param ptr The pointer that is to be checked
/// @param alignment The alignment that the pointer should be checked for
/// @return @c true if the pointer has the queried alignment, @c false otherwise
///
VECMEM_HOST
bool VECMEM_CORE_EXPORT is_aligned(void* ptr, std::size_t alignment);

}  // namespace details
}  // namespace vecmem
