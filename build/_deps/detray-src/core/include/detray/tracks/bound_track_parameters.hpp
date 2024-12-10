/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/track_parametrization.hpp"
#include "detray/definitions/units.hpp"
#include "detray/geometry/barcode.hpp"

namespace detray {

template <typename algebra_t>
struct bound_parameters_vector {

    /// @name Type definitions for the struct
    /// @{
    using algebra_type = algebra_t;
    using scalar_type = dscalar<algebra_t>;
    using point2_type = dpoint2D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using matrix_operator = dmatrix_operator<algebra_t>;

    // Underlying vector type related to bound track vector.
    using vector_type = bound_vector<algebra_t>;

    /// @}

    /// Default constructor
    bound_parameters_vector() = default;

    /// Construct from a 6-dim vector of parameters
    DETRAY_HOST_DEVICE
    explicit bound_parameters_vector(const vector_type& vec) : m_vector(vec) {}

    /// Construct from single parameters
    ///
    /// @param loc_p the bound local position
    /// @param phi the global phi angle of the track direction
    /// @param theta the global theta angle of the track direction
    /// @param qop the q/p value
    /// @param t the time
    DETRAY_HOST_DEVICE
    bound_parameters_vector(const point2_type& loc_p, const scalar_type phi,
                            const scalar_type theta, const scalar_type qop,
                            const scalar_type t) {

        matrix_operator().set_block(m_vector, loc_p, e_bound_loc0, 0u);
        matrix_operator().element(m_vector, e_bound_phi, 0u) = phi;
        matrix_operator().element(m_vector, e_bound_theta, 0u) = theta;
        matrix_operator().element(m_vector, e_bound_qoverp, 0u) = qop;
        matrix_operator().element(m_vector, e_bound_time, 0u) = t;
    }

    /// @param rhs is the left hand side params for comparison
    DETRAY_HOST_DEVICE
    bool operator==(const bound_parameters_vector& rhs) const {

        for (unsigned int i = 0u; i < e_bound_size; i++) {
            const auto lhs_val = matrix_operator().element(m_vector, i, 0u);
            const auto rhs_val = matrix_operator().element(rhs.vector(), i, 0u);

            if (math::fabs(lhs_val - rhs_val) >
                std::numeric_limits<scalar_type>::epsilon()) {
                return false;
            }
        }

        return true;
    }

    /// Convenience access to the track parameters - const
    DETRAY_HOST_DEVICE
    scalar_type operator[](const std::size_t i) const {
        return matrix_operator().element(m_vector, static_cast<unsigned int>(i),
                                         0u);
    }

    /// Convenience access to the track parameters - non-const
    DETRAY_HOST_DEVICE
    scalar_type& operator[](const std::size_t i) {
        return matrix_operator().element(m_vector, static_cast<unsigned int>(i),
                                         0u);
    }

    /// Access the track parameters as a 6-dim vector - const
    DETRAY_HOST_DEVICE
    const vector_type& vector() const { return m_vector; }

    /// Access the track parameters as a 6-dim vector - non-const
    DETRAY_HOST_DEVICE
    vector_type& vector() { return m_vector; }

    /// Set the underlying vector
    DETRAY_HOST_DEVICE
    void set_vector(const vector_type& v) { m_vector = v; }

    /// @returns the bound local position
    DETRAY_HOST_DEVICE
    point2_type bound_local() const {
        return {matrix_operator().element(m_vector, e_bound_loc0, 0u),
                matrix_operator().element(m_vector, e_bound_loc1, 0u)};
    }

    /// Set the bound local position
    DETRAY_HOST_DEVICE
    void set_bound_local(const point2_type& pos) {
        matrix_operator().set_block(m_vector, pos, e_bound_loc0, 0u);
    }

    /// @returns the global phi angle
    DETRAY_HOST_DEVICE
    scalar_type phi() const {
        return matrix_operator().element(m_vector, e_bound_phi, 0u);
    }

    /// Set the global phi angle
    DETRAY_HOST_DEVICE
    void set_phi(const scalar_type phi) {
        assert(math::fabs(phi) <= constant<scalar_type>::pi);
        matrix_operator().element(m_vector, e_bound_phi, 0u) = phi;
    }

    /// @returns the global theta angle
    DETRAY_HOST_DEVICE
    scalar_type theta() const {
        return matrix_operator().element(m_vector, e_bound_theta, 0u);
    }

    /// Set the global theta angle
    DETRAY_HOST_DEVICE
    void set_theta(const scalar_type theta) {
        assert(0.f < theta);
        assert(theta <= constant<scalar_type>::pi);
        matrix_operator().element(m_vector, e_bound_theta, 0u) = theta;
    }

    /// @returns the global track direction
    DETRAY_HOST_DEVICE
    vector3_type dir() const {
        const scalar_type phi{
            matrix_operator().element(m_vector, e_bound_phi, 0u)};
        const scalar_type theta{
            matrix_operator().element(m_vector, e_bound_theta, 0u)};
        const scalar_type sinTheta{math::sin(theta)};

        return {math::cos(phi) * sinTheta, math::sin(phi) * sinTheta,
                math::cos(theta)};
    }

    /// @returns the time
    DETRAY_HOST_DEVICE
    scalar_type time() const {
        return matrix_operator().element(m_vector, e_bound_time, 0u);
    }

    /// Set the time
    DETRAY_HOST_DEVICE
    void set_time(const scalar_type t) {
        matrix_operator().element(m_vector, e_bound_time, 0u) = t;
    }

    /// @returns the q/p value
    DETRAY_HOST_DEVICE
    scalar_type qop() const {
        return matrix_operator().element(m_vector, e_bound_qoverp, 0u);
    }

    /// Set the q/p value
    DETRAY_HOST_DEVICE
    void set_qop(const scalar_type qop) {
        matrix_operator().element(m_vector, e_bound_qoverp, 0u) = qop;
    }

    /// @returns the q/p_T value
    DETRAY_HOST_DEVICE
    scalar_type qopT() const {
        const scalar_type theta{
            matrix_operator().element(m_vector, e_bound_theta, 0u)};
        const scalar_type sinTheta{math::sin(theta)};
        assert(sinTheta != 0.f);
        return matrix_operator().element(m_vector, e_bound_qoverp, 0u) /
               sinTheta;
    }

    /// @returns the q/p_z value
    DETRAY_HOST_DEVICE
    scalar_type qopz() const {
        const scalar_type theta{
            matrix_operator().element(m_vector, e_bound_theta, 0u)};
        const scalar_type cosTheta{math::cos(theta)};
        assert(cosTheta != 0.f);
        return matrix_operator().element(m_vector, e_bound_qoverp, 0u) /
               cosTheta;
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
        assert(qop() != 0.f);
        assert(q * qop() > 0.f);
        return math::fabs(q / qop() * getter::perp(dir()));
    }

    /// @returns the absolute momentum z-component
    DETRAY_HOST_DEVICE
    scalar_type pz(const scalar_type q) const {
        assert(qop() != 0.f);
        assert(q * qop() > 0.f);
        return math::fabs(q / qop() * dir()[2]);
    }

    private:
    vector_type m_vector = matrix_operator().template zero<e_bound_size, 1>();
};

/// Combine the bound track parameter vector with the covariance and associated
/// surface
template <typename algebra_t>
struct bound_track_parameters : public bound_parameters_vector<algebra_t> {

    using base_type = bound_parameters_vector<algebra_t>;

    /// @name Type definitions for the struct
    /// @{
    using algebra_type = algebra_t;
    using scalar_type = dscalar<algebra_t>;
    using point2_type = dpoint2D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using matrix_operator = dmatrix_operator<algebra_t>;

    // Shorthand vector/matrix types related to bound track parameters.
    using parameter_vector_type = bound_parameters_vector<algebra_t>;
    using covariance_type = bound_matrix<algebra_t>;

    /// @}

    /// Default constructor sets the covaraicne to zero
    bound_track_parameters() = default;

    DETRAY_HOST_DEVICE
    bound_track_parameters(const geometry::barcode sf_idx,
                           const parameter_vector_type& vec,
                           const covariance_type& cov)
        : base_type(vec), m_covariance(cov), m_barcode(sf_idx) {}

    /// @param rhs is the left hand side params for comparison
    DETRAY_HOST_DEVICE
    bool operator==(const bound_track_parameters& rhs) const {
        if (m_barcode != rhs.surface_link()) {
            return false;
        }

        if (!base_type::operator==(rhs)) {
            return false;
        }

        for (unsigned int i = 0u; i < e_bound_size; i++) {
            for (unsigned int j = 0u; j < e_bound_size; j++) {
                const auto lhs_val =
                    matrix_operator().element(m_covariance, i, j);
                const auto rhs_val =
                    matrix_operator().element(rhs.covariance(), i, j);

                if (math::fabs(lhs_val - rhs_val) >
                    std::numeric_limits<scalar_type>::epsilon()) {
                    return false;
                }
            }
        }

        return true;
    }

    /// @returns the barcode of the associated surface
    DETRAY_HOST_DEVICE
    const geometry::barcode& surface_link() const { return m_barcode; }

    /// Set the barcode of the associated surface
    DETRAY_HOST_DEVICE
    void set_surface_link(geometry::barcode link) { m_barcode = link; }

    /// Set the track parameter vector
    DETRAY_HOST_DEVICE
    void set_parameter_vector(const parameter_vector_type& v) {
        this->set_vector(v.vector());
    }

    /// @returns the track parameter covariance - non-const
    DETRAY_HOST_DEVICE
    covariance_type& covariance() { return m_covariance; }

    /// @returns the track parameter covariance - const
    DETRAY_HOST_DEVICE
    const covariance_type& covariance() const { return m_covariance; }

    /// Set the track parameter covariance
    DETRAY_HOST_DEVICE
    void set_covariance(const covariance_type& c) { m_covariance = c; }

    private:
    covariance_type m_covariance =
        matrix_operator().template zero<e_bound_size, e_bound_size>();
    geometry::barcode m_barcode{};
};

}  // namespace detray
