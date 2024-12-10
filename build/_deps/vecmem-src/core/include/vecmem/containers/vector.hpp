/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/memory/polymorphic_allocator.hpp"

// System include(s).
#include <vector>

/// @brief Main namespace for the vecmem classes/functions
///
/// Public classes and functions that are not language/backend specific, are
/// generally placed in this namespace.
///
/// @see @c vecmem::data
///
namespace vecmem {
/**
 * @brief Alias type for vectors with our polymorphic allocator
 *
 * This type serves as an alias for a common type pattern, namely a
 * host-accessible vector with a memory resource which is not known at
 * compile time, which could be host memory or shared memory.
 *
 * @warning This type should only be used with host-accessible memory
 * resources.
 */
template <typename T>
using vector = std::vector<T, vecmem::polymorphic_allocator<T>>;

/// Helper function creating a @c vecmem::data::vector_view object
template <typename TYPE, typename ALLOC>
VECMEM_HOST data::vector_view<TYPE> get_data(std::vector<TYPE, ALLOC>& vec);

/// Helper function creating a @c vecmem::data::vector_view object
template <typename TYPE, typename ALLOC>
VECMEM_HOST data::vector_view<const TYPE> get_data(
    const std::vector<TYPE, ALLOC>& vec);

}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/vector.ipp"
