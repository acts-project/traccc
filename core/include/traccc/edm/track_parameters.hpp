/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/utils/unit_vectors.hpp"

namespace traccc {

struct TRACCC_ALIGN(32) bound_track_parameters {
    using vector_t = bound_vector;
    using covariance_t = bound_matrix;
    using jacobian_t = bound_matrix;

    vector_t m_vector;
    covariance_t m_covariance;

    // surface id
    unsigned int surface_id;

    bound_track_parameters() = default;

    TRACCC_HOST_DEVICE
    scalar charge() const {
        if (m_vector[e_bound_qoverp] < 0) {
            return -1.;
        } else {
            return 1.;
        }
    }

    TRACCC_HOST_DEVICE
    vector_t& vector() { return m_vector; }

    TRACCC_HOST_DEVICE
    const vector_t& vector() const { return m_vector; }

    TRACCC_HOST_DEVICE
    covariance_t& covariance() { return m_covariance; }

    TRACCC_HOST_DEVICE
    vector2 local() {
        return vector2({m_vector[e_bound_loc0], m_vector[e_bound_loc1]});
    }

    template <typename surface_t>
    TRACCC_HOST_DEVICE auto position(
        vecmem::vector<surface_t>& surfaces) const {
        // vector2 loc({m_vector[e_bound_loc0], m_vector[e_bound_loc1]});
        vector2 loc;
        loc[0] = m_vector[e_bound_loc0];
        loc[1] = m_vector[e_bound_loc1];

        vector3 global = surfaces.items[surface_id].local_to_global(loc);
        return global;
    }

    TRACCC_HOST_DEVICE
    vector3 unit_direction() const {
        // Need to use algebra plugin
        vector3 dir = make_direction_unit_from_phi_theta(
            m_vector[e_bound_phi], m_vector[e_bound_theta]);
        return dir;
    }

    TRACCC_HOST_DEVICE
    scalar& time() { return m_vector[e_bound_time]; }

    TRACCC_HOST_DEVICE
    scalar& qop() { return m_vector[e_bound_qoverp]; }

    template <typename surface_t>
    TRACCC_HOST_DEVICE auto reference_surface(
        vecmem::vector<surface_t>& surfaces) const {
        return surfaces.items[surface_id];
    }
};

inline bool operator==(const bound_track_parameters& lhs,
                       const bound_track_parameters& rhs) {
    const auto& vec_a = lhs.vector();
    const auto& vec_b = rhs.vector();

    if (std::abs(vec_a[e_bound_loc0] - vec_b[e_bound_loc0]) < float_epsilon &&
        std::abs(vec_a[e_bound_loc1] - vec_b[e_bound_loc1]) < float_epsilon &&
        std::abs(vec_a[e_bound_theta] - vec_b[e_bound_theta]) < float_epsilon &&
        std::abs(vec_a[e_bound_phi] - vec_b[e_bound_phi]) < float_epsilon) {
        return true;
    }
    return false;
}

/// Declare all track_parameters collection types
using bound_track_parameters_collection_types =
    collection_types<bound_track_parameters>;

struct curvilinear_track_parameters {
    using vector_t = bound_vector;
    using covariance_t = bound_sym_matrix;
    using jacobian_t = bound_matrix;

    vector_t m_vector;
    covariance_t m_covariance;

    TRACCC_HOST_DEVICE
    vector_t& vector() { return m_vector; }

    TRACCC_HOST_DEVICE
    covariance_t& covariance() { return m_covariance; }
};

struct free_track_parameters {
    using vector_t = free_vector;
    using covariance_t = free_sym_matrix;
    using jacobian_t = free_matrix;

    vector_t m_vector;
    covariance_t m_covariance;

    free_track_parameters() = default;

    free_track_parameters(scalar tx, scalar ty, scalar tz, scalar tt,
                          scalar tpx, scalar tpy, scalar tpz, scalar q) {
        m_vector[e_free_pos0] = tx;
        m_vector[e_free_pos1] = ty;
        m_vector[e_free_pos2] = tz;
        m_vector[e_free_time] = tt;
        vector3 mom({tpx, tpy, tpz});
        // scalar p = mom.norm();
        // auto mom_norm = mom.normalized();
        scalar p = getter::norm(mom);
        auto mom_norm = vector::normalize(mom);
        m_vector[e_free_dir0] = mom_norm[0];
        m_vector[e_free_dir1] = mom_norm[1];
        m_vector[e_free_dir2] = mom_norm[2];
        m_vector[e_free_qoverp] = q / p;
    }

    TRACCC_HOST_DEVICE
    vector_t& vector() { return m_vector; }

    TRACCC_HOST_DEVICE
    vector3 pos() {
        return {m_vector[e_free_pos0], m_vector[e_free_pos1],
                m_vector[e_free_pos2]};
    }

    TRACCC_HOST_DEVICE
    vector3 dir() {
        return {m_vector[e_free_dir0], m_vector[e_free_dir1],
                m_vector[e_free_dir2]};
    }

    TRACCC_HOST_DEVICE
    covariance_t& covariance() { return m_covariance; }

    TRACCC_HOST_DEVICE
    scalar& qop() { return m_vector[e_free_qoverp]; }
};

}  // namespace traccc
