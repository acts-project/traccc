/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/binary_page_memory_resource.hpp"

#include "details/binary_page_memory_resource_impl.hpp"
#include "details/memory_resource_impl.hpp"

namespace vecmem {

binary_page_memory_resource::binary_page_memory_resource(
    memory_resource& upstream)
    : m_impl{std::make_unique<details::binary_page_memory_resource_impl>(
          upstream)} {}

VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(binary_page_memory_resource)

}  // namespace vecmem
