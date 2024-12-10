/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/contiguous_memory_resource.hpp"

#include "details/contiguous_memory_resource_impl.hpp"
#include "details/memory_resource_impl.hpp"

namespace vecmem {

contiguous_memory_resource::contiguous_memory_resource(
    memory_resource& upstream, std::size_t size)
    : m_impl{std::make_unique<details::contiguous_memory_resource_impl>(
          upstream, size)} {}

VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(contiguous_memory_resource)

}  // namespace vecmem
