/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Covfie include(s).
#include <covfie/core/concepts.hpp>
#include <covfie/core/field.hpp>

// System include(s).
#include <any>

namespace traccc {

/// Typeless, owning, host-only magnetic field object
class magnetic_field {

    public:
    /// Default constructor
    magnetic_field() = default;

    /// Constructor from a specific b-field object
    ///
    /// @tparam bfield_backend_t The backend type of the b-field object
    /// @param obj The b-field object to construct from
    ///
    template <covfie::concepts::field_backend bfield_backend_t>
    explicit magnetic_field(covfie::field<bfield_backend_t>&& obj);

    /// Set a specific b-field object
    ///
    /// @tparam bfield_backend_t The backend type of the b-field object
    /// @param obj The b-field object to set
    ///
    template <covfie::concepts::field_backend bfield_backend_t>
    void set(covfie::field<bfield_backend_t>&& obj);

    /// Check if the b-field is of a certain type
    ///
    /// @tparam bfield_backend_t The covfie backend type to check
    /// @return @c true if the b-field is of the specified type,
    ///         @c false otherwise
    ///
    template <covfie::concepts::field_backend bfield_backend_t>
    bool is() const;

    /// Get a b-field view object as a specific type
    ///
    /// @tparam bfield_backend_t The covfie backend type to use
    /// @return The b-field view object of the specified type
    ///
    template <covfie::concepts::field_backend bfield_backend_t>
    typename covfie::field<bfield_backend_t>::view_t as_view() const;

    /// Get the b-field object as a specific type
    ///
    /// @tparam bfield_backend_t The covfie backend type to use
    /// @return The b-field object cast to the specified type
    ///
    template <covfie::concepts::field_backend bfield_backend_t>
    const covfie::field<bfield_backend_t>& as_field() const;

    private:
    /// The actualy covfie b-field object
    std::any m_field;

};  // class magnetic_field

}  // namespace traccc

// Include the implementation.
#include "traccc/bfield/impl/magnetic_field.ipp"
