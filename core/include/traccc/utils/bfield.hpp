/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/primitives.hpp"

// Covfie include(s).
#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/clamp.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

// System include(s).
#include <any>
#include <sstream>

namespace traccc {

/// Typeless, owning, host-only magnetic field object
class bfield {

    public:
    /// Default constructor
    bfield() = default;

    /// Constructor from a specific b-field object
    ///
    /// @tparam bfield_backend_t The backend type of the b-field object
    /// @param obj The b-field object to construct from
    ///
    template <typename bfield_backend_t>
    explicit bfield(covfie::field<bfield_backend_t>&& obj) requires(
        covfie::concepts::field_backend<bfield_backend_t>)
        : m_field(std::move(obj)) {}

    /// Set a specific b-field object
    ///
    /// @tparam bfield_backend_t The backend type of the b-field object
    /// @param obj The b-field object to set
    ///
    template <typename bfield_backend_t>
    void set(covfie::field<bfield_backend_t>&& obj) requires(
        covfie::concepts::field_backend<bfield_backend_t>) {
        m_field = std::move(obj);
    }

    /// Check if the b-field is of a certain type
    ///
    /// @tparam bfield_backend_t The covfie backend type to check
    /// @return @c true if the b-field is of the specified type,
    ///         @c false otherwise
    ///
    template <typename bfield_backend_t>
    bool is() const
        requires(covfie::concepts::field_backend<bfield_backend_t>) {
        return (m_field.type() == typeid(covfie::field<bfield_backend_t>));
    }

    /// @brief Return type information about the contained magnetic field.
    const std::type_info& type() const { return m_field.type(); }

    /// Get a b-field view object as a specific type
    ///
    /// @tparam bfield_backend_t The covfie backend type to use
    /// @return The b-field view object of the specified type
    ///
    template <typename bfield_backend_t>
    typename covfie::field<bfield_backend_t>::view_t as() const
        requires(covfie::concepts::field_backend<bfield_backend_t>) {
        return typename covfie::field<bfield_backend_t>::view_t{
            std::any_cast<const covfie::field<bfield_backend_t>&>(m_field)};
    }

    /// Get the b-field object as a specific type
    ///
    /// @tparam bfield_backend_t The covfie backend type to use
    /// @return The b-field object cast to the specified type
    ///
    template <typename bfield_backend_t>
    const covfie::field<bfield_backend_t>& get_covfie_field() const
        requires(covfie::concepts::field_backend<bfield_backend_t>) {
        return std::any_cast<const covfie::field<bfield_backend_t>&>(m_field);
    }

    private:
    /// The actualy covfie b-field object
    std::any m_field;

};  // class bfield

/// Constant magnetic field backend type
template <typename scalar_t>
using const_bfield_backend_t =
    ::covfie::backend::constant<::covfie::vector::vector_d<scalar_t, 3>,
                                ::covfie::vector::vector_d<scalar_t, 3>>;

/// Inhomogeneous magnetic field used for IO
template <typename scalar_t>
using inhom_io_bfield_backend_t =
    covfie::backend::affine<covfie::backend::linear<covfie::backend::strided<
        covfie::vector::vector_d<std::size_t, 3>,
        covfie::backend::array<covfie::vector::vector_d<scalar_t, 3>>>>>;

/// Inhomogeneous magnetic field backend type
template <typename scalar_t>
using inhom_bfield_backend_t = covfie::backend::affine<
    covfie::backend::linear<covfie::backend::clamp<covfie::backend::strided<
        covfie::vector::vector_d<std::size_t, 3>,
        covfie::backend::array<covfie::vector::vector_d<scalar_t, 3>>>>>>;

/// Construct a constant magnetic field object
template <typename scalar_t>
::covfie::field<const_bfield_backend_t<scalar_t>> construct_const_bfield(
    scalar_t x, scalar_t y, scalar_t z) {
    return ::covfie::field<const_bfield_backend_t<scalar_t>>{
        ::covfie::make_parameter_pack(
            typename const_bfield_backend_t<scalar_t>::configuration_t{x, y,
                                                                       z})};
}

/// Construct a constant magnetic field object
template <typename scalar_t>
::covfie::field<const_bfield_backend_t<scalar_t>> construct_const_bfield(
    const vector3& v) {
    return construct_const_bfield(v[0], v[1], v[2]);
}

/// @brief The standard list of host bfield types to support
template <typename scalar_t>
using host_bfield_type_list = std::tuple<const_bfield_backend_t<scalar_t>,
                                         inhom_bfield_backend_t<scalar_t>>;

/// @brief Helper function for `bfield_visitor`
template <typename callable_t, typename bfield_t, typename... bfield_ts>
auto bfield_visitor_helper(
    const bfield& bfield, callable_t&& callable,
    std::tuple<
        bfield_t,
        bfield_ts...>*) requires(covfie::concepts::field_backend<bfield_t> &&
                                 (covfie::concepts::field_backend<bfield_ts> &&
                                  ...)) {
    if (bfield.is<bfield_t>()) {
        return callable(bfield.as<bfield_t>());
    } else {
        if constexpr (sizeof...(bfield_ts) > 0) {
            return bfield_visitor_helper(
                bfield, std::forward<callable_t>(callable),
                static_cast<std::tuple<bfield_ts...>*>(nullptr));
        } else {
            std::stringstream exception_message;

            exception_message
                << "Invalid B-field type (" << bfield.type().name()
                << ") received, but this type is not supported" << std::endl;

            throw std::invalid_argument(exception_message.str());
        }
    }
}

/// @brief Visitor for polymorphic magnetic field types
///
/// This function takes a list of supported magnetic field types and checks
/// if the provided field is one of them. If it is, it will call the provided
/// callable on a view of it and otherwise it will throw an exception.
template <typename bfield_list_t, typename callable_t>
auto bfield_visitor(const bfield& bfield, callable_t&& callable) {
    return bfield_visitor_helper(bfield, std::forward<callable_t>(callable),
                                 static_cast<bfield_list_t*>(nullptr));
}
}  // namespace traccc
