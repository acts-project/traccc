/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/utils/tuple.hpp"
#if __cplusplus >= 201700L
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/vector.hpp"
#endif  // __cplusplus >= 201700L

// System include(s).
#include <tuple>
#include <type_traits>

namespace vecmem {
namespace edm {
namespace details {

/// @brief The type of the proxy to use for a given container variable
enum class proxy_type {
    /// Proxy for a host container element
    host,
    /// Proxy for a device container element
    device
};

/// @brief The "access type" of the proxy to use for a given container variable
enum class proxy_access {
    /// Proxy for a non-const container element
    non_constant,
    /// Proxy for a const container element
    constant
};

/// @name Traits for the proxied variable types
/// @{

/// Technical base class for the @c proxy_var_type traits
template <typename VTYPE, proxy_type PTYPE, proxy_access CTYPE>
struct proxy_var_type;

/// Constant access to a scalar variable (both host and device)
template <typename VTYPE, proxy_type PTYPE>
struct proxy_var_type<type::scalar<VTYPE>, PTYPE, proxy_access::constant> {

    /// The scalar is kept by value in the proxy
    using type = std::add_lvalue_reference_t<std::add_const_t<VTYPE>>;
    /// It is returned as a const reference even on non-const access
    using return_type = type;
    /// It is returned as a const reference on const access
    using const_return_type = return_type;

    /// Helper function constructing a scalar proxy variable
    template <typename ITYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE, return_type variable) {
        return variable;
    }
};

/// Non-const access to a scalar variable (both host and device)
template <typename VTYPE, proxy_type PTYPE>
struct proxy_var_type<type::scalar<VTYPE>, PTYPE, proxy_access::non_constant> {

    /// The scalar is kept by lvalue reference in the proxy
    using type = std::add_lvalue_reference_t<VTYPE>;
    /// It is returned as a non-const lvalue reference on non-const access
    using return_type = type;
    /// It is returned as a const reference on const access
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<VTYPE>>;

    /// Helper function constructing a scalar proxy variable
    template <typename ITYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE, return_type variable) {
        return variable;
    }
};

/// Constant access to a vector variable (both host and device)
template <typename VTYPE, proxy_type PTYPE>
struct proxy_var_type<type::vector<VTYPE>, PTYPE, proxy_access::constant> {

    /// Vector elements are kept by value in the proxy
    using type = std::add_lvalue_reference_t<std::add_const_t<VTYPE>>;
    /// They are returned as a const reference even on non-const access
    using return_type = type;
    /// They are returned as a const reference on const access
    using const_return_type = return_type;

    /// Helper function constructing a vector proxy variable
    template <typename ITYPE, typename VECTYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE i, const VECTYPE& vec) {

        return vec.at(i);
    }
};

/// Non-const access to a vector variable (both host and device)
template <typename VTYPE, proxy_type PTYPE>
struct proxy_var_type<type::vector<VTYPE>, PTYPE, proxy_access::non_constant> {

    /// Vector elements are kept by lvalue reference in the proxy
    using type = std::add_lvalue_reference_t<VTYPE>;
    /// They are returned as a non-const lvalue reference on non-const access
    using return_type = type;
    /// They are returned as a const reference on const access
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<VTYPE>>;

    /// Helper function constructing a vector proxy variable
    template <typename ITYPE, typename VECTYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE i, VECTYPE& vec) {

        return vec.at(i);
    }
};

/// Constant access to a jagged vector variable from a device container
template <typename VTYPE>
struct proxy_var_type<type::jagged_vector<VTYPE>, proxy_type::device,
                      proxy_access::constant> {

    /// Jagged vector elements are kept by constant device vectors in the proxy
    using type = device_vector<std::add_const_t<VTYPE>>;
    /// They are returned as a const reference to the device vector even in
    /// non-const access
    using return_type = std::add_lvalue_reference_t<std::add_const_t<type>>;
    /// They are returned as a const reference to the device vector on const
    /// access
    using const_return_type = return_type;

    /// Helper function constructing a vector proxy variable
    VECMEM_HOST_AND_DEVICE
    static type make(
        typename jagged_device_vector<std::add_const_t<VTYPE>>::size_type i,
        const jagged_device_vector<std::add_const_t<VTYPE>>& vec) {

        return vec.at(i);
    }
};

/// Non-const access to a jagged vector variable from a device container
template <typename VTYPE>
struct proxy_var_type<type::jagged_vector<VTYPE>, proxy_type::device,
                      proxy_access::non_constant> {

    /// Jagged vector elements are kept by non-const device vectors in the proxy
    using type = device_vector<VTYPE>;
    /// They are returned as non-const lvalue references to the non-const device
    /// vector in non-const access
    using return_type = std::add_lvalue_reference_t<type>;
    /// They are returned as const references to the non-const device vector in
    /// const access
    using const_return_type = std::add_lvalue_reference_t<
        std::add_const_t<device_vector<std::add_const_t<VTYPE>>>>;

    /// Helper function constructing a vector proxy variable
    VECMEM_HOST_AND_DEVICE
    static type make(typename jagged_device_vector<VTYPE>::size_type i,
                     jagged_device_vector<VTYPE>& vec) {

        return vec.at(i);
    }
};

#if __cplusplus >= 201700L

/// Constant access to a jagged vector variable from a host container
template <typename VTYPE>
struct proxy_var_type<type::jagged_vector<VTYPE>, proxy_type::host,
                      proxy_access::constant> {

    /// Jagged vector elements are kept by constant reference in the proxy
    using type = std::add_lvalue_reference_t<std::add_const_t<vector<VTYPE>>>;
    /// They are returned as a const reference even on non-const access
    using return_type = type;
    /// They are returned as a const reference on const access
    using const_return_type = type;

    /// Helper function constructing a vector proxy variable
    VECMEM_HOST
    static type make(typename jagged_vector<VTYPE>::size_type i,
                     const jagged_vector<VTYPE>& vec) {

        return vec.at(i);
    }
};

/// Non-const access to a jagged vector variable from a host container
template <typename VTYPE>
struct proxy_var_type<type::jagged_vector<VTYPE>, proxy_type::host,
                      proxy_access::non_constant> {

    /// Jagged vector elements are kept by non-const lvalue reference in the
    /// proxy
    using type = std::add_lvalue_reference_t<vector<VTYPE>>;
    /// They are returned as a non-const lvalue reference on non-const access
    using return_type = type;
    /// They are returned as a const reference on const access
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<vector<VTYPE>>>;

    /// Helper function constructing a vector proxy variable
    VECMEM_HOST
    static type make(typename jagged_vector<VTYPE>::size_type i,
                     jagged_vector<VTYPE>& vec) {

        return vec.at(i);
    }
};

#endif  // __cplusplus >= 201700L

/// Proxy types for one element of a type pack
template <std::size_t INDEX, proxy_type PTYPE, proxy_access CTYPE,
          typename... VARTYPES>
struct proxy_var_type_at {
    /// Type of the variable held by the proxy
    using type =
        typename proxy_var_type<tuple_element_t<INDEX, tuple<VARTYPES...>>,
                                PTYPE, CTYPE>::type;
    /// Return type on non-const access to the proxy
    using return_type =
        typename proxy_var_type<tuple_element_t<INDEX, tuple<VARTYPES...>>,
                                PTYPE, CTYPE>::return_type;
    /// Return type on const access to the proxy
    using const_return_type =
        typename proxy_var_type<tuple_element_t<INDEX, tuple<VARTYPES...>>,
                                PTYPE, CTYPE>::const_return_type;
};

/// @}

/// @name Traits for creating the proxy data tuples
/// @{

/// Technical base class for the @c proxy_data_creator traits
template <typename SCHEMA, proxy_type PTYPE, proxy_access CTYPE>
struct proxy_data_creator;

/// Helper class making the data tuple for a constant device proxy
template <typename VARTYPE, proxy_type PTYPE>
struct proxy_data_creator<schema<VARTYPE>, PTYPE, proxy_access::constant> {

    /// Make all other instantiations of the struct friends
    template <typename, proxy_type, proxy_access>
    friend struct proxy_data_creator;

    /// Proxy tuple type created by the helper
    using proxy_tuple_type = tuple<
        typename proxy_var_type<VARTYPE, PTYPE, proxy_access::constant>::type>;

    /// Construct the tuple used by the proxy
    template <typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make(ITYPE i,
                                                        const CONTAINER& c) {
        return make_impl<0>(i, c);
    }

private:
    template <std::size_t INDEX, typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make_impl(
        ITYPE i, const CONTAINER& c) {

        return {proxy_var_type<VARTYPE, PTYPE, proxy_access::constant>::make(
            i, c.template get<INDEX>())};
    }
};

/// Helper class making the data tuple for a non-const device proxy
template <typename VARTYPE, proxy_type PTYPE>
struct proxy_data_creator<schema<VARTYPE>, PTYPE, proxy_access::non_constant> {

    /// Make all other instantiations of the struct friends
    template <typename, proxy_type, proxy_access>
    friend struct proxy_data_creator;

    /// Proxy tuple type created by the helper
    using proxy_tuple_type =
        tuple<typename proxy_var_type<VARTYPE, PTYPE,
                                      proxy_access::non_constant>::type>;

    /// Construct the tuple used by the proxy
    template <typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make(ITYPE i, CONTAINER& c) {
        return make_impl<0>(i, c);
    }

private:
    template <std::size_t INDEX, typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make_impl(ITYPE i,
                                                             CONTAINER& c) {

        return {
            proxy_var_type<VARTYPE, PTYPE, proxy_access::non_constant>::make(
                i, c.template get<INDEX>())};
    }
};

/// Helper class making the data tuple for a constant device proxy
template <typename VARTYPE, typename... VARTYPES, proxy_type PTYPE>
struct proxy_data_creator<schema<VARTYPE, VARTYPES...>, PTYPE,
                          proxy_access::constant> {

    /// Make all other instantiations of the struct friends
    template <typename, proxy_type, proxy_access>
    friend struct proxy_data_creator;

    /// Proxy tuple type created by the helper
    using proxy_tuple_type = tuple<
        typename proxy_var_type<VARTYPE, PTYPE, proxy_access::constant>::type,
        typename proxy_var_type<VARTYPES, PTYPE,
                                proxy_access::constant>::type...>;

    /// Construct the tuple used by the proxy
    template <typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make(ITYPE i,
                                                        const CONTAINER& c) {
        return make_impl<0>(i, c);
    }

private:
    template <std::size_t INDEX, typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make_impl(
        ITYPE i, const CONTAINER& c) {

        return proxy_tuple_type(
            proxy_var_type<VARTYPE, PTYPE, proxy_access::constant>::make(
                i, c.template get<INDEX>()),
            proxy_data_creator<
                schema<VARTYPES...>, PTYPE,
                proxy_access::constant>::template make_impl<INDEX + 1>(i, c));
    }
};

/// Helper class making the data tuple for a non-const device proxy
template <typename VARTYPE, typename... VARTYPES, proxy_type PTYPE>
struct proxy_data_creator<schema<VARTYPE, VARTYPES...>, PTYPE,
                          proxy_access::non_constant> {

    /// Make all other instantiations of the struct friends
    template <typename, proxy_type, proxy_access>
    friend struct proxy_data_creator;

    /// Proxy tuple type created by the helper
    using proxy_tuple_type =
        tuple<typename proxy_var_type<VARTYPE, PTYPE,
                                      proxy_access::non_constant>::type,
              typename proxy_var_type<VARTYPES, PTYPE,
                                      proxy_access::non_constant>::type...>;

    /// Construct the tuple used by the proxy
    template <typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make(ITYPE i, CONTAINER& c) {
        return make_impl<0>(i, c);
    }

private:
    template <std::size_t INDEX, typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make_impl(ITYPE i,
                                                             CONTAINER& c) {

        return proxy_tuple_type(
            proxy_var_type<VARTYPE, PTYPE, proxy_access::non_constant>::make(
                i, c.template get<INDEX>()),
            proxy_data_creator<
                schema<VARTYPES...>, PTYPE,
                proxy_access::non_constant>::template make_impl<INDEX + 1>(i,
                                                                           c));
    }
};

/// @}

}  // namespace details
}  // namespace edm
}  // namespace vecmem
