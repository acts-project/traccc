/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/edm/schema.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {
namespace edm {
namespace details {

/// @name Traits for the view types for the individual variables
/// @{

template <typename TYPE>
struct view_type_base {
    using payload_type = TYPE;
    using payload_ptr = std::add_pointer_t<payload_type>;
    using size_type = unsigned int;
    using size_ptr = std::add_pointer_t<size_type>;
};  // struct view_type_base

template <typename TYPE>
struct view_type : public view_type_base<TYPE> {};

template <typename TYPE>
struct view_type<type::scalar<TYPE> > : public view_type_base<TYPE> {
    using layout_type = int;
    using layout_ptr = std::add_pointer_t<layout_type>;
    using type = typename view_type_base<TYPE>::payload_ptr;
};  // struct view_type

template <typename TYPE>
struct view_type<type::vector<TYPE> > : public view_type_base<TYPE> {
    using layout_type = int;
    using layout_ptr = std::add_pointer_t<layout_type>;
    using type = vecmem::data::vector_view<TYPE>;
};  // struct view_type

template <typename TYPE>
struct view_type<type::jagged_vector<TYPE> > : public view_type_base<TYPE> {
    using layout_type = vecmem::data::vector_view<TYPE>;
    using layout_ptr = std::add_pointer_t<layout_type>;
    using type = vecmem::data::jagged_vector_view<TYPE>;
};  // struct view_type

/// @}

}  // namespace details
}  // namespace edm
}  // namespace vecmem
