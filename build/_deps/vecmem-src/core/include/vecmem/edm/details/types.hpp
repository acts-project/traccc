/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {
namespace edm {
namespace details {

/// Type-less, non-const view of a memory block
using memory_view = data::vector_view<char>;

/// Type-less, const view of a memory block
using const_memory_view = data::vector_view<const char>;

/// Size type used in the SoA classes
using size_type = memory_view::size_type;

/// Non-const pointer type to the size of the SoA classes
using size_pointer = std::add_pointer_t<size_type>;

/// Constant pointer type to the size of the SoA classes
using const_size_pointer = std::add_pointer_t<std::add_const_t<size_type> >;

}  // namespace details
}  // namespace edm
}  // namespace vecmem
