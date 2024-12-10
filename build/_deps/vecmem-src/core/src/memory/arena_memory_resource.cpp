/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/arena_memory_resource.hpp"

#include "details/arena_memory_resource_impl.hpp"
#include "details/memory_resource_impl.hpp"

namespace vecmem {

arena_memory_resource::arena_memory_resource(memory_resource& upstream,
                                             std::size_t initial_size,
                                             std::size_t maximum_size)
    : m_impl{std::make_unique<details::arena_memory_resource_impl>(
          initial_size, maximum_size, upstream)} {}

VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(arena_memory_resource)

}  // namespace vecmem
