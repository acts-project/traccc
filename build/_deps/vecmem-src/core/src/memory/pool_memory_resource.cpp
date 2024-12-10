/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/pool_memory_resource.hpp"

#include "details/memory_resource_impl.hpp"
#include "details/pool_memory_resource_impl.hpp"

namespace vecmem {

pool_memory_resource::options::options() = default;

pool_memory_resource::pool_memory_resource(memory_resource& upstream,
                                           const options& opts)
    : m_impl(std::make_unique<details::pool_memory_resource_impl>(upstream,
                                                                  opts)) {}

VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(pool_memory_resource)

}  // namespace vecmem
