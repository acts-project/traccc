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
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

// System include(s).
#include <any>

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
    explicit bfield(covfie::field<bfield_backend_t>&& obj)
        : m_field(std::move(obj)) {}

    /// Set a specific b-field object
    ///
    /// @tparam bfield_backend_t The backend type of the b-field object
    /// @param obj The b-field object to set
    ///
    template <typename bfield_backend_t>
    void set(covfie::field<bfield_backend_t>&& obj) {
        m_field = std::move(obj);
    }

    /// Check if the b-field is of a certain type
    ///
    /// @tparam bfield_backend_t The covfie backend type to check
    /// @return @c true if the b-field is of the specified type,
    ///         @c false otherwise
    ///
    template <typename bfield_backend_t>
    bool is() const {
        return (m_field.type() == typeid(covfie::field<bfield_backend_t>));
    }

    /// Get a b-field view object as a specific type
    ///
    /// @tparam bfield_backend_t The covfie backend type to use
    /// @return The b-field view object of the specified type
    ///
    template <typename bfield_backend_t>
    typename covfie::field<bfield_backend_t>::view_t as() const {
        return typename covfie::field<bfield_backend_t>::view_t{
            std::any_cast<const covfie::field<bfield_backend_t>&>(m_field)};
    }

    /// Get the b-field object as a specific type
    ///
    /// @tparam bfield_backend_t The covfie backend type to use
    /// @return The b-field object cast to the specified type
    ///
    template <typename bfield_backend_t>
    const covfie::field<bfield_backend_t>& get_covfie_field() const {
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
}  // namespace traccc
