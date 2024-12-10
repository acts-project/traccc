/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/conditional_memory_resource.hpp"

#include "details/conditional_memory_resource_impl.hpp"
#include "details/memory_resource_impl.hpp"

namespace vecmem {

conditional_memory_resource::conditional_memory_resource(
    memory_resource& upstream,
    std::function<bool(std::size_t, std::size_t)> pred)
    : m_impl{std::make_unique<details::conditional_memory_resource_impl>(
          upstream, pred)} {}

VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(conditional_memory_resource)

}  // namespace vecmem
