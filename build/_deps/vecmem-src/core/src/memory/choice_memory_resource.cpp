/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/choice_memory_resource.hpp"

#include "details/choice_memory_resource_impl.hpp"
#include "details/memory_resource_impl.hpp"

namespace vecmem {

choice_memory_resource::choice_memory_resource(
    std::function<memory_resource&(std::size_t, std::size_t)> decision)
    : m_impl{std::make_unique<details::choice_memory_resource_impl>(decision)} {
}

VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(choice_memory_resource)

}  // namespace vecmem
