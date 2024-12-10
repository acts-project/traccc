/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_data.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/types.hpp"

namespace vecmem {

/**
 * @brief Alias type for jagged vectors with our polymorphic allocator
 *
 * This type serves as an alias for a common type pattern, namely a
 * host-accessible vector of vectors with a memory resource which is not
 * known at compile time, which could be host memory or shared memory.
 *
 * @warning This type should only be used with host-accessible memory
 * resources.
 */
template <typename T>
using jagged_vector = vector<vector<T>>;

/// Helper function creating a @c vecmem::data::jagged_vector_data object
template <typename TYPE>
VECMEM_HOST data::jagged_vector_data<TYPE> get_data(
    jagged_vector<TYPE>& vec, memory_resource* resource = nullptr);

/// Helper function creating a @c vecmem::data::jagged_vector_data object
template <typename TYPE, typename ALLOC1, typename ALLOC2>
VECMEM_HOST data::jagged_vector_data<TYPE> get_data(
    std::vector<std::vector<TYPE, ALLOC1>, ALLOC2>& vec,
    memory_resource* resource);

/// Helper function creating a @c vecmem::data::jagged_vector_data object
template <typename TYPE>
VECMEM_HOST data::jagged_vector_data<const TYPE> get_data(
    const jagged_vector<TYPE>& vec, memory_resource* resource = nullptr);

/// Helper function creating a @c vecmem::data::jagged_vector_data object
template <typename TYPE, typename ALLOC1, typename ALLOC2>
VECMEM_HOST data::jagged_vector_data<const TYPE> get_data(
    const std::vector<std::vector<TYPE, ALLOC1>, ALLOC2>& vec,
    memory_resource* resource);

}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/jagged_vector.ipp"
