/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/coalescing_memory_resource.hpp"

#include "details/coalescing_memory_resource_impl.hpp"
#include "details/memory_resource_impl.hpp"

namespace vecmem {

coalescing_memory_resource::coalescing_memory_resource(
    std::vector<std::reference_wrapper<memory_resource>>&& upstreams)
    : m_impl{std::make_unique<details::coalescing_memory_resource_impl>(
          std::move(upstreams))} {}

VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(coalescing_memory_resource)

}  // namespace vecmem
