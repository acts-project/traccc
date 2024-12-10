/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/edm/details/types.hpp"
#include "vecmem/edm/details/view_traits.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/utils/tuple.hpp"
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {

/// Namespace for the types implementing Struct-of-Array container support
namespace edm {

/// Technical base type for @c view<schema<VARTYPES...>>
template <typename T>
class view;

/// View of a Struct-of-Arrays container
///
/// Much like how @c vecmem::data::vector_view can be used to communicate the
/// needed information about a single 1D vector to a device function, this type
/// is meant to communicate the same type of information about an entire
/// container of variables.
///
/// @tparam ...VARTYPES The variable types described by the view
///
template <typename... VARTYPES>
class view<schema<VARTYPES...>> {

    // Sanity check(s).
    static_assert(sizeof...(VARTYPES) > 0,
                  "SoA containers without variables are not supported.");

public:
    /// The schema describing the container view
    using schema_type = schema<VARTYPES...>;
    /// Size type used for the container
    using size_type = details::size_type;
    /// Pointer type to the size of the container
    using size_pointer = std::conditional_t<
        vecmem::details::disjunction_v<std::is_const<
            typename details::view_type<VARTYPES>::payload_type>...>,
        details::const_size_pointer, details::size_pointer>;
    /// Constant pointer type to the size of the container
    using const_size_pointer = details::const_size_pointer;
    /// The tuple type holding all of the views for the individual variables
    using tuple_type = tuple<typename details::view_type<VARTYPES>::type...>;
    /// Type of the view(s) into the raw data of the view
    using memory_view_type = std::conditional_t<
        vecmem::details::disjunction_v<std::is_const<
            typename details::view_type<VARTYPES>::payload_type>...>,
        details::const_memory_view, details::memory_view>;

    /// @name Constructors and assignment operators
    /// @{

    /// Default constructor
    view() = default;

    /// Constructor with a capacity and size.
    ///
    /// @param capacity The maximum capacity of the container
    /// @param size Optional pointer to the size of the container
    ///
    VECMEM_HOST_AND_DEVICE
    view(size_type capacity, const memory_view_type& size = {0u, nullptr});

    /// Constructor from a (possibly/slightly) different view
    ///
    /// As with @c vecmem::data::vector_view and
    /// @c vecmem::data::jagged_vector_view, this constructor must only be
    /// active for non-const to const conversions. As we need to use the default
    /// copy and move constructors for copying/moving identical types. Otherwise
    /// SYCL is not happy with sending these as kernel parameters.
    ///
    /// @tparam OTHERTYPES The variable types described by the "other view"
    /// @param other The "other view" to copy from
    ///
    template <
        typename... OTHERTYPES,
        std::enable_if_t<
            vecmem::details::conjunction_v<std::is_constructible<
                typename details::view_type<VARTYPES>::type,
                typename details::view_type<OTHERTYPES>::type>...> &&
                vecmem::details::disjunction_v<
                    vecmem::details::negation<std::is_same<
                        typename details::view_type<VARTYPES>::type,
                        typename details::view_type<OTHERTYPES>::type>>...>,
            bool> = true>
    VECMEM_HOST_AND_DEVICE view(const view<schema<OTHERTYPES...>>& other);

    /// Assignment operator from a (possibly/slightly) different view
    ///
    /// @see view(const view<schema<OTHERTYPES...>>&)
    ///
    /// @tparam OTHERTYPES The variable types described by the "other view"
    /// @param rhs The "other view" to assign from
    ///
    template <
        typename... OTHERTYPES,
        std::enable_if_t<
            vecmem::details::conjunction_v<std::is_constructible<
                typename details::view_type<VARTYPES>::type,
                typename details::view_type<OTHERTYPES>::type>...> &&
                vecmem::details::disjunction_v<
                    vecmem::details::negation<std::is_same<
                        typename details::view_type<VARTYPES>::type,
                        typename details::view_type<OTHERTYPES>::type>>...>,
            bool> = true>
    VECMEM_HOST_AND_DEVICE view& operator=(
        const view<schema<OTHERTYPES...>>& rhs);

    /// @}

    /// @name Function(s) meant for normal, client use
    /// @{

    /// Get the maximum capacity of the container
    VECMEM_HOST_AND_DEVICE
    size_type capacity() const;

    /// Get the view of a specific variable (non-const)
    template <std::size_t INDEX>
    VECMEM_HOST_AND_DEVICE tuple_element_t<INDEX, tuple_type>& get();
    /// Get the view of a specific variable (const)
    template <std::size_t INDEX>
    VECMEM_HOST_AND_DEVICE const tuple_element_t<INDEX, tuple_type>& get()
        const;

    /// @}

    /// @name Function(s) meant for internal use by other VecMem types
    /// @{

    /// Direct (non-const) access to the underlying tuple of views
    VECMEM_HOST_AND_DEVICE
    tuple_type& variables();
    /// Direct (const) access to the underlying tuple of views
    VECMEM_HOST_AND_DEVICE
    const tuple_type& variables() const;

    /// View of the memory allocated for the container's size variable(s)
    VECMEM_HOST_AND_DEVICE
    const memory_view_type& size() const;

    /// View at the single (device) memory allocation of the container
    VECMEM_HOST_AND_DEVICE
    const memory_view_type& payload() const;

    /// View at the memory that describes the layout of the container
    VECMEM_HOST_AND_DEVICE
    const memory_view_type& layout() const;
    /// View at the memory that describes the layout of the container (in host
    /// accessible memory)
    VECMEM_HOST_AND_DEVICE
    const memory_view_type& host_layout() const;

    /// @}

protected:
    /// Maximum capacity of the container
    size_type m_capacity;
    /// Views for the individual variables
    tuple_type m_views;

    /// View into the memory allocated for the container's size variable(s)
    memory_view_type m_size;

    /// View into the single (device) memory allocation for the "payload"
    memory_view_type m_payload;

    /// View into the memory that describes the layout of the container
    memory_view_type m_layout;
    /// View into the memory that describes the layout of the container (in host
    /// accessible memory)
    memory_view_type m_host_layout;

};  // class view

}  // namespace edm
}  // namespace vecmem

// Include the implementation.
#include "vecmem/edm/impl/view.ipp"
