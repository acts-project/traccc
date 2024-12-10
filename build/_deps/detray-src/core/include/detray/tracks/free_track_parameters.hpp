/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/math.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/track_parametrization.hpp"

namespace detray {

template <typename algebra_t>
struct free_parameters_vector {

    /// @name Type definitions for the struct
    /// @{
    using algebra_type = algebra_t;
    using scalar_type = dscalar<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using matrix_operator = dmatrix_operator<algebra_t>;

    // Shorthand vector type related to free track parameters.
    using vector_type = free_vector<algebra_t>;

    /// @}

    /// Default constructor
    free_parameters_vector() = default;

    /// Construct from a 6-dim vector of parameters
    DETRAY_HOST_DEVICE
    explicit free_parameters_vector(const vector_type& vec) : m_vector(vec) {}

    /// Construct from single parameters
    ///
    /// @param pos the global position
    /// @param time the time
    /// @param mom the global track momentum 3-vector
    /// @param q the particle charge
    DETRAY_HOST_DEVICE
    free_parameters_vector(const point3_type& pos, const scalar_type time,
                           const vector3_type& mom, const scalar_type q) {

        matrix_operator().set_block(m_vector, pos, e_free_pos0, 0u);
        matrix_operator().element(m_vector, e_free_time, 0u) = time;

        scalar_type p = getter::norm(mom);
        auto mom_norm = vector::normalize(mom);
        matrix_operator().set_block(m_vector, mom_norm, e_free_dir0, 0u);
        matrix_operator().element(m_vector, e_free_qoverp, 0u) = q / p;
    }

    /// @param rhs is the left hand side params for comparison
    DETRAY_HOST_DEVICE
    bool operator==(const free_parameters_vector& rhs) const {
        for (unsigned int i = 0u; i < e_free_size; i++) {
            if (math::fabs((*this)[i] - rhs[i]) >
                std::numeric_limits<scalar_type>::epsilon()) {
                return false;
            }
        }

        return true;
    }

    /// Convenience access to the track parameters - const
    DETRAY_HOST_DEVICE
    scalar_type operator[](std::size_t i) const {
        return matrix_operator().element(m_vector, static_cast<unsigned int>(i),
                                         0u);
    }

    /// Convenience access to the track parameters - non-const
    DETRAY_HOST_DEVICE
    scalar_type& operator[](std::size_t i) {
        return matrix_operator().element(m_vector, static_cast<unsigned int>(i),
                                         0u);
    }

    /// @returns the global track position
    DETRAY_HOST_DEVICE
    point3_type pos() const {
        return {matrix_operator().element(m_vector, e_free_pos0, 0u),
                matrix_operator().element(m_vector, e_free_pos1, 0u),
                matrix_operator().element(m_vector, e_free_pos2, 0u)};
    }

    /// Set the global track position
    DETRAY_HOST_DEVICE
    void set_pos(const vector3_type& pos) {
        matrix_operator().set_block(m_vector, pos, e_free_pos0, 0u);
    }

    /// @returns the normalized, global track direction
    DETRAY_HOST_DEVICE
    vector3_type dir() const {
        return {matrix_operator().element(m_vector, e_free_dir0, 0u),
                matrix_operator().element(m_vector, e_free_dir1, 0u),
                matrix_operator().element(m_vector, e_free_dir2, 0u)};
    }

    /// Set the global track direction
    /// @note Must be normalized!
    DETRAY_HOST_DEVICE
    void set_dir(const vector3_type& dir) {
        matrix_operator().set_block(m_vector, dir, e_free_dir0, 0u);
    }

    /// @returns the time
    DETRAY_HOST_DEVICE
    scalar_type time() const {
        return matrix_operator().element(m_vector, e_free_time, 0u);
    }

    /// Set the time
    DETRAY_HOST_DEVICE
    void set_time(const scalar_type t) {
        matrix_operator().element(m_vector, e_free_time, 0u) = t;
    }

    /// @returns the q/p value
    DETRAY_HOST_DEVICE
    scalar_type qop() const {
        return matrix_operator().element(m_vector, e_free_qoverp, 0u);
    }

    /// Set the q/p value
    DETRAY_HOST_DEVICE
    void set_qop(const scalar_type qop) {
        matrix_operator().element(m_vector, e_free_qoverp, 0u) = qop;
    }

    /// @returns the q/p_T value
    DETRAY_HOST_DEVICE
    scalar_type qopT() const {
        const auto dir = this->dir();
        assert(getter::perp(dir) != 0.f);
        return matrix_operator().element(m_vector, e_free_qoverp, 0u) /
               getter::perp(dir);
    }

    /// @returns the q/p_z value
    DETRAY_HOST_DEVICE
    scalar_type qopz() const {
        const auto dir = this->dir();
        return matrix_operator().element(m_vector, e_free_qoverp, 0u) / dir[2];
    }

    /// @returns the absolute momentum
    DETRAY_HOST_DEVICE
    scalar_type p(const scalar_type q) const {
        assert(qop() != 0.f);
        assert(q * qop() > 0.f);
        return q / qop();
    }

    /// @returns the global momentum 3-vector
    DETRAY_HOST_DEVICE
    vector3_type mom(const scalar_type q) const { return p(q) * dir(); }

    /// @returns the transverse momentum
    DETRAY_HOST_DEVICE
    scalar_type pT(const scalar_type q) const {
        assert(this->qop() != 0.f);
        assert(q * qop() > 0.f);
        return math::fabs(q / this->qop() * getter::perp(this->dir()));
    }

    /// @returns the absolute momentum z-component
    DETRAY_HOST_DEVICE
    scalar_type pz(const scalar_type q) const {
        assert(this->qop() != 0.f);
        assert(q * qop() > 0.f);
        return math::fabs(q / this->qop() * this->dir()[2]);
    }

    private:
    vector_type m_vector = matrix_operator().template zero<e_free_size, 1>();
};

/// The free track parameters consist only of the parameter vector
template <typename algebra_t>
using free_track_parameters = free_parameters_vector<algebra_t>;

}  // namespace detray
