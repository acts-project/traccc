/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/debug_memory_resource.hpp"

#include "details/debug_memory_resource_impl.hpp"
#include "details/memory_resource_impl.hpp"

namespace vecmem {

debug_memory_resource::debug_memory_resource(memory_resource& upstream)
    : m_impl{std::make_unique<details::debug_memory_resource_impl>(upstream)} {}

VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(debug_memory_resource)

}  // namespace vecmem
