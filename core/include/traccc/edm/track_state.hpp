/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"

// detray include(s).
#include "detray/navigation/navigator.hpp"
#include "detray/tracks/bound_track_parameters.hpp"

namespace traccc {

/// Fitting result per track
template <typename algebra_t>
struct fitting_result {
    using scalar_type = detray::dscalar<algebra_t>;

    /// Fitted track parameter
    detray::bound_track_parameters<algebra_t> fit_params;

    /// Number of degree of freedoms of fitted track
    scalar_type ndf{0};

    /// Chi square of fitted track
    scalar_type chi2{0};

    // The number of holes (The number of sensitive surfaces which do not have a
    // measurement for the track pattern)
    unsigned int n_holes{0u};
};

/// Fitting result per measurement
template <typename algebra_t>
struct track_state {

    using scalar_type = detray::dscalar<algebra_t>;

    using bound_track_parameters_type =
        detray::bound_track_parameters<algebra_t>;
    using bound_matrix = detray::bound_matrix<algebra_t>;
    using matrix_operator = detray::dmatrix_operator<algebra_t>;
    using size_type = detray::dsize_type<algebra_t>;
    template <size_type ROWS, size_type COLS>
    using matrix_type = detray::dmatrix<algebra_t, ROWS, COLS>;

    track_state() = default;

    /// Construction with track candidate
    TRACCC_HOST_DEVICE
    track_state(const track_candidate& trk_cand)
        : m_surface_link(trk_cand.surface_link), m_measurement(trk_cand) {
        m_predicted.set_surface_link(m_surface_link);
        m_filtered.set_surface_link(m_surface_link);
        m_smoothed.set_surface_link(m_surface_link);
    }

    /// @return the surface link
    TRACCC_HOST_DEVICE
    inline detray::geometry::barcode surface_link() const {
        return m_surface_link;
    }

    /// @return the measurement
    TRACCC_HOST_DEVICE
    inline const measurement& get_measurement() const { return m_measurement; }

    /// @return the local position of measurement with 2 X 1 matrix
    // FIXME: The conversion from vector to matrix is inefficient
    template <size_type D>
    TRACCC_HOST_DEVICE inline matrix_type<D, 1> measurement_local() const {
        static_assert(((D == 1u) || (D == 2u)),
                      "The measurement dimension should be 1 or 2");

        matrix_type<D, 1> ret;
        if (m_measurement.subs.get_indices()[0] == e_bound_loc0) {
            matrix_operator().element(ret, 0, 0) = m_measurement.local[0];
            if constexpr (D == 2u) {
                matrix_operator().element(ret, 1, 0) = m_measurement.local[1];
            }
        } else if (m_measurement.subs.get_indices()[0] == e_bound_loc1) {
            matrix_operator().element(ret, 0, 0) = m_measurement.local[1];
            if constexpr (D == 2u) {
                matrix_operator().element(ret, 1, 0) = m_measurement.local[0];
            }
        } else {
            assert(
                "The measurement index out of e_bound_loc0 and e_bound_loc1 "
                "should not happen.");
            matrix_operator().element(ret, 0, 0) = m_measurement.local[0];
            if constexpr (D == 2u) {
                matrix_operator().element(ret, 1, 0) = m_measurement.local[1];
            }
        }

        return ret;
    }

    /// @return the covariance of local position of measurement
    template <size_type D>
    TRACCC_HOST_DEVICE inline matrix_type<D, D> measurement_covariance() const {
        static_assert(((D == 1u) || (D == 2u)),
                      "The measurement dimension should be 1 or 2");

        matrix_type<D, D> ret;
        if (m_measurement.subs.get_indices()[0] == e_bound_loc0) {

            matrix_operator().element(ret, 0, 0) = m_measurement.variance[0];
            if constexpr (D == 2u) {
                matrix_operator().element(ret, 0, 1) = 0.f;
                matrix_operator().element(ret, 1, 0) = 0.f;
                matrix_operator().element(ret, 1, 1) =
                    m_measurement.variance[1];
            }

        } else if (m_measurement.subs.get_indices()[0] == e_bound_loc1) {

            matrix_operator().element(ret, 0, 0) = m_measurement.variance[1];
            if constexpr (D == 2u) {
                matrix_operator().element(ret, 0, 1) = 0.f;
                matrix_operator().element(ret, 1, 0) = 0.f;
                matrix_operator().element(ret, 1, 1) =
                    m_measurement.variance[0];
            }
        } else {
            assert(
                "The measurement index out of e_bound_loc0 and e_bound_loc1 "
                "should not happen.");
            matrix_operator().element(ret, 0, 0) = m_measurement.variance[0];
            if constexpr (D == 2u) {
                matrix_operator().element(ret, 0, 1) = 0.f;
                matrix_operator().element(ret, 1, 0) = 0.f;
                matrix_operator().element(ret, 1, 1) =
                    m_measurement.variance[1];
            }
        }
        return ret;
    }

    /// @return the non-const reference of predicted track state
    TRACCC_HOST_DEVICE
    inline bound_track_parameters_type& predicted() { return m_predicted; }

    /// @return the const reference of predicted track state
    TRACCC_HOST_DEVICE
    inline const bound_track_parameters_type& predicted() const {
        return m_predicted;
    }

    /// @return the non-const transport jacobian
    TRACCC_HOST_DEVICE
    inline bound_matrix& jacobian() { return m_jacobian; }

    /// @return the const transport jacobian
    TRACCC_HOST_DEVICE
    inline const bound_matrix& jacobian() const { return m_jacobian; }

    /// @return the non-const chi square of filtered parameter
    TRACCC_HOST_DEVICE
    inline scalar_type& filtered_chi2() { return m_filtered_chi2; }

    /// @return the const chi square of filtered parameter
    TRACCC_HOST_DEVICE
    inline const scalar_type& filtered_chi2() const { return m_filtered_chi2; }

    /// @return the non-const filtered parameter
    TRACCC_HOST_DEVICE
    inline bound_track_parameters_type& filtered() { return m_filtered; }

    /// @return the const filtered parameter
    TRACCC_HOST_DEVICE
    inline const bound_track_parameters_type& filtered() const {
        return m_filtered;
    }

    /// @return the non-const chi square of smoothed parameter
    TRACCC_HOST_DEVICE
    inline scalar_type& smoothed_chi2() { return m_smoothed_chi2; }

    /// @return the const chi square of smoothed parameter
    TRACCC_HOST_DEVICE
    inline const scalar_type& smoothed_chi2() const { return m_smoothed_chi2; }

    /// @return the non-const smoothed parameter
    TRACCC_HOST_DEVICE
    inline bound_track_parameters_type& smoothed() { return m_smoothed; }

    /// @return the const smoothed parameter
    TRACCC_HOST_DEVICE
    inline const bound_track_parameters_type& smoothed() const {
        return m_smoothed;
    }

    public:
    bool is_hole{true};

    private:
    detray::geometry::barcode m_surface_link;
    measurement m_measurement;
    bound_matrix m_jacobian =
        matrix_operator().template zero<e_bound_size, e_bound_size>();
    bound_track_parameters_type m_predicted;
    scalar_type m_filtered_chi2 = 0.f;
    bound_track_parameters_type m_filtered;
    scalar_type m_smoothed_chi2 = 0.f;
    bound_track_parameters_type m_smoothed;
};

/// Declare all track_state collection types
using track_state_collection_types =
    collection_types<track_state<default_algebra>>;

/// Declare all track_state container types
using track_state_container_types =
    container_types<fitting_result<default_algebra>,
                    track_state<default_algebra>>;

}  // namespace traccc
