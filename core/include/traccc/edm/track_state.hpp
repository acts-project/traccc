/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_quality.hpp"

// detray include(s).
#include <detray/tracks/bound_track_parameters.hpp>

namespace traccc {

enum class fitter_outcome : uint32_t {
    UNKNOWN,
    SUCCESS,
    FAILURE_NON_POSITIVE_NDF,
    FAILURE_NOT_ALL_SMOOTHED,
    MAX_OUTCOME
};

/// Fitting result per track
template <typename algebra_t>
struct fitting_result {
    using scalar_type = detray::dscalar<algebra_t>;

    /// Fitting outcome
    fitter_outcome fit_outcome = fitter_outcome::UNKNOWN;

    /// Fitted track parameter
    traccc::bound_track_parameters<algebra_t> fit_params;

    /// Track quality
    traccc::track_quality trk_quality;
};

/// Fitting result per measurement
template <typename algebra_t>
struct track_state {

    using scalar_type = detray::dscalar<algebra_t>;
    using size_type = detray::dsize_type<algebra_t>;

    using bound_track_parameters_type =
        traccc::bound_track_parameters<algebra_t>;
    using bound_matrix_type = traccc::bound_matrix<algebra_t>;
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
        assert((m_measurement.subs.get_indices()[0] == e_bound_loc0) ||
               (m_measurement.subs.get_indices()[0] == e_bound_loc1));

        matrix_type<D, 1> ret;
        switch (m_measurement.subs.get_indices()[0]) {
            case e_bound_loc0:
                getter::element(ret, 0, 0) = m_measurement.local[0];
                if constexpr (D == 2u) {
                    getter::element(ret, 1, 0) = m_measurement.local[1];
                }
                break;
            case e_bound_loc1:
                getter::element(ret, 0, 0) = m_measurement.local[1];
                if constexpr (D == 2u) {
                    getter::element(ret, 1, 0) = m_measurement.local[0];
                }
                break;
            default:
                __builtin_unreachable();
        }
        return ret;
    }

    /// @return the covariance of local position of measurement
    template <size_type D>
    TRACCC_HOST_DEVICE inline matrix_type<D, D> measurement_covariance() const {
        static_assert(((D == 1u) || (D == 2u)),
                      "The measurement dimension should be 1 or 2");
        assert((m_measurement.subs.get_indices()[0] == e_bound_loc0) ||
               (m_measurement.subs.get_indices()[0] == e_bound_loc1));

        matrix_type<D, D> ret;
        switch (m_measurement.subs.get_indices()[0]) {
            case e_bound_loc0:
                getter::element(ret, 0, 0) = m_measurement.variance[0];
                if constexpr (D == 2u) {
                    getter::element(ret, 0, 1) = 0.f;
                    getter::element(ret, 1, 0) = 0.f;
                    getter::element(ret, 1, 1) = m_measurement.variance[1];
                }
                break;
            case e_bound_loc1:
                getter::element(ret, 0, 0) = m_measurement.variance[1];
                if constexpr (D == 2u) {
                    getter::element(ret, 0, 1) = 0.f;
                    getter::element(ret, 1, 0) = 0.f;
                    getter::element(ret, 1, 1) = m_measurement.variance[0];
                }
                break;
            default:
                __builtin_unreachable();
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
    inline bound_matrix_type& jacobian() { return m_jacobian; }

    /// @return the const transport jacobian
    TRACCC_HOST_DEVICE
    inline const bound_matrix_type& jacobian() const { return m_jacobian; }

    /// @return the non-const chi square of filtered parameter
    TRACCC_HOST_DEVICE
    inline scalar_type& filtered_chi2() { return m_filtered_chi2; }

    /// @return the const chi square of filtered parameter
    TRACCC_HOST_DEVICE
    inline const scalar_type& filtered_chi2() const { return m_filtered_chi2; }

    /// @return the non-const chi square of backward filter
    TRACCC_HOST_DEVICE
    inline scalar_type& backward_chi2() { return m_backward_chi2; }

    /// @return the const chi square of backward filter
    TRACCC_HOST_DEVICE
    inline scalar_type backward_chi2() const { return m_backward_chi2; }

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
    bool is_smoothed{false};

    private:
    detray::geometry::barcode m_surface_link;
    measurement m_measurement;
    bound_matrix_type m_jacobian = matrix::zero<bound_matrix_type>();
    bound_track_parameters_type m_predicted;
    scalar_type m_filtered_chi2 = 0.f;
    bound_track_parameters_type m_filtered;
    scalar_type m_smoothed_chi2 = 0.f;
    bound_track_parameters_type m_smoothed;
    scalar_type m_backward_chi2 = 0.f;
};

/// Declare all track_state collection types
using track_state_collection_types =
    collection_types<track_state<default_algebra>>;

/// Declare all track_state container types
using track_state_container_types =
    container_types<fitting_result<default_algebra>,
                    track_state<default_algebra>>;

inline std::size_t count_fitted_tracks(
    const track_state_container_types::host& track_states) {

    const std::size_t n_tracks = track_states.size();
    std::size_t n_fitted_tracks = 0u;

    for (std::size_t i = 0; i < n_tracks; i++) {
        if (track_states.at(i).header.fit_outcome == fitter_outcome::SUCCESS) {
            n_fitted_tracks++;
        }
    }

    return n_fitted_tracks;
}

}  // namespace traccc
