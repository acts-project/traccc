/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
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
#include "detray/propagator/navigator.hpp"
#include "detray/tracks/bound_track_parameters.hpp"

namespace traccc {

/// Fitting result per track
template <typename algebra_t>
struct fitter_info {
    using scalar_type = typename algebra_t::scalar_type;

    /// Fitted track parameter
    detray::bound_track_parameters<algebra_t> fit_params;

    /// Number of degree of freedoms of fitted track
    scalar_type ndf{0};

    /// Chi square of fitted track
    scalar_type chi2{0};
};

/// Fitting result per measurement
template <typename algebra_t>
struct track_state {

    using bound_track_parameters_type =
        detray::bound_track_parameters<algebra_t>;
    using bound_matrix = typename bound_track_parameters_type::covariance_type;
    using scalar_type = typename algebra_t::scalar_type;
    using matrix_operator = typename algebra_t::matrix_actor;
    using size_type = typename matrix_operator::size_ty;
    template <size_type ROWS, size_type COLS>
    using matrix_type =
        typename matrix_operator::template matrix_type<ROWS, COLS>;

    track_state() = default;

    /// Construction with track candidate
    TRACCC_HOST_DEVICE
    track_state(const track_candidate& trk_cand)
        : m_surface_link(trk_cand.surface_link), m_measurement(trk_cand.meas) {
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
    TRACCC_HOST_DEVICE
    inline matrix_type<2, 1> measurement_local() const {
        matrix_type<2, 1> ret = matrix_operator().template zero<2, 1>();
        matrix_operator().element(ret, 0, 0) = m_measurement.local[0];
        matrix_operator().element(ret, 1, 0) = m_measurement.local[1];
        return ret;
    }

    /// @return the covariance of local position of measurement
    TRACCC_HOST_DEVICE
    inline matrix_type<2, 2> measurement_covariance() const {
        matrix_type<2, 2> ret = matrix_operator().template zero<2, 2>();
        matrix_operator().element(ret, 0, 0) = m_measurement.variance[0];
        matrix_operator().element(ret, 1, 1) = m_measurement.variance[1];
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
    scalar_type m_filtered_chi2;
    bound_track_parameters_type m_filtered;
    scalar_type m_smoothed_chi2;
    bound_track_parameters_type m_smoothed;
};

/// Declare all track_state collection types
using track_state_collection_types = collection_types<track_state<transform3>>;

/// Declare all track_state container types
using track_state_container_types =
    container_types<fitter_info<transform3>, track_state<transform3>>;

}  // namespace traccc