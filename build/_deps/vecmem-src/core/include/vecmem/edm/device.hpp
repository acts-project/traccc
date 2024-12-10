/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/device_traits.hpp"
#include "vecmem/edm/proxy.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/edm/view.hpp"
#include "vecmem/utils/tuple.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <utility>

namespace vecmem {
namespace edm {

/// Technical base type for @c device<schema<VARTYPES...>,INTERFACE>
template <typename T, template <typename> class I>
class device;

/// Structure-of-Arrays device container
///
/// This class implements a structure-of-arrays container using device vector
/// types. Allowing the same operations on the individual variables that are
/// available from @c vecmem::device_vector and @c vecmem::jagged_device_vector.
///
/// @tparam ...VARTYPES The variable types stored in the container
///
template <typename... VARTYPES, template <typename> class INTERFACE>
class device<schema<VARTYPES...>, INTERFACE> {

    // Sanity check(s).
    static_assert(sizeof...(VARTYPES) > 0,
                  "SoA containers without variables are not supported.");

public:
    /// The schema describing the device-accessible variables
    using schema_type = schema<VARTYPES...>;
    /// Size type used for the container
    using size_type = typename view<schema_type>::size_type;
    /// Pointer type to the size of the container
    using size_pointer = typename view<schema_type>::size_pointer;
    /// The tuple type holding all of the the individual "device objects"
    using tuple_type = tuple<typename details::device_type<VARTYPES>::type...>;
    /// The type of the interface used for the proxy objects
    template <typename T>
    using interface_type = INTERFACE<T>;
    /// The type of the (non-const) proxy objects for the container elements
    using proxy_type =
        interface_type<proxy<schema_type, details::proxy_type::device,
                             details::proxy_access::non_constant>>;
    /// The type of the (const) proxy objects for the container elements
    using const_proxy_type =
        interface_type<proxy<schema_type, details::proxy_type::device,
                             details::proxy_access::constant>>;

    /// @name Constructors and assignment operators
    /// @{

    /// Constructor from an approptiate view
    VECMEM_HOST_AND_DEVICE
    device(const view<schema_type>& view);

    /// @}

    /// @name Function(s) meant for normal, client use
    /// @{

    /// Get the size of the container
    VECMEM_HOST_AND_DEVICE
    size_type size() const;
    /// Get the maximum capacity of the container
    VECMEM_HOST_AND_DEVICE
    size_type capacity() const;

    /// Add one default element to all (vector) variables (thread safe)
    VECMEM_HOST_AND_DEVICE
    size_type push_back_default();

    /// Get a specific variable (non-const)
    template <std::size_t INDEX>
    VECMEM_HOST_AND_DEVICE
        typename details::device_type_at<INDEX, VARTYPES...>::return_type
        get();
    /// Get a specific variable (const)
    template <std::size_t INDEX>
    VECMEM_HOST_AND_DEVICE
        typename details::device_type_at<INDEX, VARTYPES...>::const_return_type
        get() const;

    /// Get a proxy object for a specific variable (non-const)
    ///
    /// While also checking that the index is within bounds.
    ///
    /// @param index The index of the element to proxy
    /// @return A proxy object for the element at the given index
    ///
    VECMEM_HOST_AND_DEVICE
    proxy_type at(size_type index);
    /// Get a proxy object for a specific variable (const)
    ///
    /// While also checking that the index is within bounds.
    ///
    /// @param index The index of the element to proxy
    /// @return A proxy object for the element at the given index
    ///
    VECMEM_HOST_AND_DEVICE
    const_proxy_type at(size_type index) const;

    /// Get a proxy object for a specific variable (non-const)
    ///
    /// @param index The index of the element to proxy
    /// @return A proxy object for the element at the given index
    ///
    VECMEM_HOST_AND_DEVICE
    proxy_type operator[](size_type index);
    /// Get a proxy object for a specific variable (const)
    ///
    /// @param index The index of the element to proxy
    /// @return A proxy object for the element at the given index
    ///
    VECMEM_HOST_AND_DEVICE
    const_proxy_type operator[](size_type index) const;

    /// @}

    /// @name Function(s) meant for internal use by other VecMem types
    /// @{

    /// Direct (non-const) access to the underlying tuple of variables
    VECMEM_HOST_AND_DEVICE
    tuple_type& variables();
    /// Direct (const) access to the underlying tuple of variables
    VECMEM_HOST_AND_DEVICE
    const tuple_type& variables() const;

    /// @}

private:
    /// Construct a default element for every vector variable
    template <std::size_t INDEX, std::size_t... Is>
    VECMEM_HOST_AND_DEVICE void construct_default(
        size_type index, std::index_sequence<INDEX, Is...>);
    /// Construct a default element for every vector variable (terminal node)
    VECMEM_HOST_AND_DEVICE void construct_default(size_type index,
                                                  std::index_sequence<>);

    /// Default, no-op vector element construction helper function
    template <typename T>
    VECMEM_HOST_AND_DEVICE void construct_vector(size_type, T&);
    /// Vector element constructor helper function
    template <typename T>
    VECMEM_HOST_AND_DEVICE void construct_vector(size_type index,
                                                 device_vector<T>& vec);

    /// Maximum capacity of the container
    size_type m_capacity = 0;
    /// (Resizable) Size of the container described by this view
    size_pointer m_size = nullptr;
    /// The tuple holding all of the individual "device objects"
    tuple_type m_data;

};  // class device

}  // namespace edm
}  // namespace vecmem

// Include the implementation.
#include "vecmem/edm/impl/device.ipp"
