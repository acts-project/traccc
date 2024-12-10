/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/data.hpp"
#include "vecmem/edm/details/host_traits.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/proxy.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <functional>
#include <tuple>

namespace vecmem {
namespace edm {

/// Technical base type for @c host<schema<VARTYPES...>,INTERFACE>
template <typename T, template <typename> class I>
class host;

/// Structure-of-Arrays host container
///
/// This class implements a structure-of-arrays container using host vector
/// types. Allowing free construction, resizing, etc. of the individual
/// variables in the client code.
///
/// @tparam ...VARTYPES The variable types to store in the host container
/// @tparam INTERFACE    The interface type to use for the container('s proxies)
///
template <typename... VARTYPES, template <typename> class INTERFACE>
class host<schema<VARTYPES...>, INTERFACE> {

public:
    /// The schema describing the host's payload
    using schema_type = schema<VARTYPES...>;
    /// Size type used for the container
    using size_type = std::size_t;
    /// The tuple type holding all of the the individual variable vectors
    using tuple_type =
        std::tuple<typename details::host_type<VARTYPES>::type...>;
    /// The type of the interface used for the proxy objects
    template <typename T>
    using interface_type = INTERFACE<T>;
    /// The type of the (non-const) proxy objects for the container elements
    using proxy_type =
        interface_type<proxy<schema_type, details::proxy_type::host,
                             details::proxy_access::non_constant>>;
    /// The type of the (const) proxy objects for the container elements
    using const_proxy_type =
        interface_type<proxy<schema_type, details::proxy_type::host,
                             details::proxy_access::constant>>;

    /// @name Constructors and assignment operators
    /// @{

    /// Constructor with a mandatory memory resource
    VECMEM_HOST
    host(memory_resource& resource);

    /// @}

    /// @name Function(s) meant for normal, client use
    /// @{

    /// Get the size of the container
    VECMEM_HOST
    size_type size() const;
    /// Resize the container
    VECMEM_HOST
    void resize(size_type size);
    /// Reserve memory for the container
    VECMEM_HOST
    void reserve(size_type size);

    /// Get the vector of a specific variable (non-const)
    template <std::size_t INDEX>
    VECMEM_HOST typename details::host_type_at<INDEX, VARTYPES...>::return_type
    get();
    /// Get the vector of a specific variable (const)
    template <std::size_t INDEX>
    VECMEM_HOST
        typename details::host_type_at<INDEX, VARTYPES...>::const_return_type
        get() const;

    /// Get a proxy object for a specific variable (non-const)
    ///
    /// While also checking that the index is within bounds.
    ///
    /// @param index The index of the element to proxy
    /// @return A proxy object for the element at the given index
    ///
    VECMEM_HOST
    proxy_type at(size_type index);
    /// Get a proxy object for a specific variable (const)
    ///
    /// While also checking that the index is within bounds.
    ///
    /// @param index The index of the element to proxy
    /// @return A proxy object for the element at the given index
    ///
    VECMEM_HOST
    const_proxy_type at(size_type index) const;

    /// Get a proxy object for a specific variable (non-const)
    ///
    /// @param index The index of the element to proxy
    /// @return A proxy object for the element at the given index
    ///
    VECMEM_HOST
    proxy_type operator[](size_type index);
    /// Get a proxy object for a specific variable (const)
    ///
    /// @param index The index of the element to proxy
    /// @return A proxy object for the element at the given index
    ///
    VECMEM_HOST
    const_proxy_type operator[](size_type index) const;

    /// @}

    /// @name Function(s) meant for internal use by other VecMem types
    /// @{

    /// Direct (non-const) access to the underlying tuple of variables
    VECMEM_HOST
    tuple_type& variables();
    /// Direct (const) access to the underlying tuple of variables
    VECMEM_HOST
    const tuple_type& variables() const;

    /// The memory resource used by the host container
    VECMEM_HOST
    memory_resource& resource() const;

    /// @}

private:
    /// The tuple holding the individual variable vectors
    tuple_type m_data;
    /// The memory resource used by the host container
    std::reference_wrapper<memory_resource> m_resource;

};  // class host

}  // namespace edm

/// Helper function for getting a (non-const) data object for a host container
///
/// @tparam ...VARTYPES The variable types describing the container
/// @param host The host container to get a data object for
/// @param resource The memory resource to use for any allocation(s)
/// @return A (non-const) data object describing the host container
///
template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST edm::data<edm::schema<VARTYPES...>> get_data(
    edm::host<edm::schema<VARTYPES...>, INTERFACE>& host,
    memory_resource* resource = nullptr);

/// Helper function for getting a (const) data object for a host container
///
/// @tparam ...VARTYPES The variable types describing the container
/// @param host The host container to get a data object for
/// @param resource The memory resource to use for any allocation(s)
/// @return A (const) data object describing the host container
///
template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST edm::data<edm::details::add_const_t<edm::schema<VARTYPES...>>>
get_data(const edm::host<edm::schema<VARTYPES...>, INTERFACE>& host,
         memory_resource* resource = nullptr);

}  // namespace vecmem

// Include the implementation.
#include "vecmem/edm/impl/host.ipp"
