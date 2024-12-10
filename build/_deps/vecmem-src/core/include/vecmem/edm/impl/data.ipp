/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/data_traits.hpp"

namespace vecmem {
namespace edm {

template <typename... VARTYPES>
VECMEM_HOST edm::data<edm::schema<VARTYPES...>>::data(size_type size,
                                                      memory_resource& resource)
    : view_type(size),
      m_data{std::make_tuple(
          details::data_alloc<VARTYPES>::make(size, resource)...)} {

    // Set up the views for the jagged vector variables correctly.
    details::data_view_assign<VARTYPES...>(
        view_type::m_views, m_data, std::index_sequence_for<VARTYPES...>());
}

template <typename... VARTYPES>
VECMEM_HOST auto edm::data<edm::schema<VARTYPES...>>::variables()
    -> tuple_type& {

    return m_data;
}

template <typename... VARTYPES>
VECMEM_HOST auto edm::data<edm::schema<VARTYPES...>>::variables() const
    -> const tuple_type& {

    return m_data;
}

}  // namespace edm

template <typename... VARTYPES>
VECMEM_HOST edm::view<edm::schema<VARTYPES...>>& get_data(
    edm::data<edm::schema<VARTYPES...>>& data) {

    return data;
}

template <typename... VARTYPES>
VECMEM_HOST edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>>
get_data(const edm::data<edm::schema<VARTYPES...>>& data) {

    return data;
}

}  // namespace vecmem
