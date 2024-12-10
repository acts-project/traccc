/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/data_traits.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/edm/view.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <tuple>

namespace vecmem {
namespace edm {

/// Technical base type for @c data<schema<VARTYPES...>>
template <typename T>
class data;

/// Data object describing a Struct-of-Arrays container
///
/// When getting "the data" for a @c vecmem::edm::host container, we can't
/// directly create a @c vecmem::edm::view object. Not if there are any jagged
/// vectors in the container. As those require the data object to hold on to
/// some memory. Exactly like @c vecmem::data::jagged_vector_data does.
///
/// This is the type created by
/// @c vecmem::get_data(vecmem::edm::host<schema<VARTYPES...>>&) calls.
/// Which then needs to be held on to by the host code, while a
/// @c vecmem::edm::view<schema<VARTYPES...>> of the data object is passed to
/// any device function that wants to use data from the host container.
///
/// @tparam ...VARTYPES The variable types to store in the data object
///
template <typename... VARTYPES>
class data<schema<VARTYPES...>> : public view<schema<VARTYPES...>> {

public:
    /// The schema describing the buffer's payload
    using schema_type = schema<VARTYPES...>;
    /// Base view type
    using view_type = view<schema_type>;
    /// Size type used for the container
    using size_type = typename view_type::size_type;
    /// The tuple type holding all of the data objects for the individual
    /// variables
    using tuple_type =
        std::tuple<typename details::data_type<VARTYPES>::type...>;

    /// @name Constructors and assignment operators
    /// @{

    /// Default constructor
    data() = default;
    /// Move constructor
    data(data&&) = default;

    /// Move assignment operator
    data& operator=(data&&) = default;

    /// Constructor for the data object
    ///
    /// @param size The size of the (outer) arrays in the data object
    /// @param resource The memory resource to use for the allocation(s)
    ///
    VECMEM_HOST
    data(size_type size, memory_resource& resource);

    /// @}

    /// @name Function(s) meant for internal use by other VecMem types
    /// @{

    /// Direct (non-const) access to the underlying tuple of data objects
    VECMEM_HOST
    tuple_type& variables();
    /// Direct (const) access to the underlying tuple of data objects
    VECMEM_HOST
    const tuple_type& variables() const;

    /// @}

private:
    /// Variable holding data for the jagged vector variables
    tuple_type m_data;

};  // class data

}  // namespace edm

/// Helper function for getting a (possibly non-const) view of a data object
///
/// @tparam ...VARTYPES The variable types describing the container
/// @param data The data object to get a view for
/// @return A (possibly non-const) view into for the data object
///
template <typename... VARTYPES>
VECMEM_HOST edm::view<edm::schema<VARTYPES...>>& get_data(
    edm::data<edm::schema<VARTYPES...>>& data);

/// Helper function for getting a (const) view of a data object
///
/// @tparam ...VARTYPES The variable types describing the container
/// @param data The data object to get a view for
/// @return A (const) view into for the data object
///
template <typename... VARTYPES>
VECMEM_HOST edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>>
get_data(const edm::data<edm::schema<VARTYPES...>>& data);

}  // namespace vecmem

// Include the implementation.
#include "vecmem/edm/impl/data.ipp"
