/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/utils/type_traits.hpp"

// System include(s).
#include <cassert>

namespace vecmem {
namespace edm {

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE view<schema<VARTYPES...>>::view(
    size_type capacity, const memory_view_type& size)
    : m_capacity(capacity),
      m_views{},
      m_size{size},
      m_payload{0, nullptr},
      m_layout{0, nullptr},
      m_host_layout{0, nullptr} {}

template <typename... VARTYPES>
template <typename... OTHERTYPES,
          std::enable_if_t<
              vecmem::details::conjunction_v<std::is_constructible<
                  typename details::view_type<VARTYPES>::type,
                  typename details::view_type<OTHERTYPES>::type>...> &&
                  vecmem::details::disjunction_v<
                      vecmem::details::negation<std::is_same<
                          typename details::view_type<VARTYPES>::type,
                          typename details::view_type<OTHERTYPES>::type>>...>,
              bool>>
VECMEM_HOST_AND_DEVICE view<schema<VARTYPES...>>::view(
    const view<schema<OTHERTYPES...>>& other)
    : m_capacity{other.capacity()},
      m_views{other.variables()},
      m_size{other.size()},
      m_payload{other.payload()},
      m_layout{other.layout()},
      m_host_layout{other.host_layout()} {}

template <typename... VARTYPES>
template <typename... OTHERTYPES,
          std::enable_if_t<
              vecmem::details::conjunction_v<std::is_constructible<
                  typename details::view_type<VARTYPES>::type,
                  typename details::view_type<OTHERTYPES>::type>...> &&
                  vecmem::details::disjunction_v<
                      vecmem::details::negation<std::is_same<
                          typename details::view_type<VARTYPES>::type,
                          typename details::view_type<OTHERTYPES>::type>>...>,
              bool>>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::operator=(
    const view<schema<OTHERTYPES...>>& rhs) -> view& {

    // Note that self-assignment with this function should never be a thing.
    // So we don't need to check for it in production code.
    assert(static_cast<const void*>(this) != static_cast<const void*>(&rhs));

    // Copy the data from the other view.
    m_capacity = rhs.capacity();
    m_views = rhs.variables();
    m_size = rhs.size();
    m_payload = rhs.payload();
    m_layout = rhs.layout();
    m_host_layout = rhs.host_layout();

    // Return a reference to this object.
    return *this;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::capacity() const
    -> size_type {

    return m_capacity;
}

template <typename... VARTYPES>
template <std::size_t INDEX>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::get()
    -> tuple_element_t<INDEX, tuple_type>& {

    return vecmem::get<INDEX>(m_views);
}

template <typename... VARTYPES>
template <std::size_t INDEX>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::get() const
    -> const tuple_element_t<INDEX, tuple_type>& {

    return vecmem::get<INDEX>(m_views);
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::variables()
    -> tuple_type& {

    return m_views;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::variables() const
    -> const tuple_type& {

    return m_views;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::size() const
    -> const memory_view_type& {

    return m_size;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::payload() const
    -> const memory_view_type& {

    return m_payload;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::layout() const
    -> const memory_view_type& {

    return m_layout;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::host_layout() const
    -> const memory_view_type& {

    return m_host_layout;
}

}  // namespace edm
}  // namespace vecmem
