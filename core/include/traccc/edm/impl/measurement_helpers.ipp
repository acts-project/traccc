/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::edm {

template <detray::concepts::algebra algebra_t, std::integral size_t, size_t D>
TRACCC_HOST_DEVICE void get_measurement_local(
    const measurement& meas, detray::dmatrix<algebra_t, D, 1>& pos) {

    static_assert(((D == 1u) || (D == 2u)),
                  "The measurement dimension must be 1 or 2");

    assert((meas.subs.get_indices()[0] == e_bound_loc0) ||
           (meas.subs.get_indices()[0] == e_bound_loc1));

    const point2& local = meas.local;

    switch (meas.subs.get_indices()[0]) {
        case e_bound_loc0:
            getter::element(pos, 0, 0) = local[0];
            if constexpr (D == 2u) {
                getter::element(pos, 1, 0) = local[1];
            }
            break;
        case e_bound_loc1:
            getter::element(pos, 0, 0) = local[1];
            if constexpr (D == 2u) {
                getter::element(pos, 1, 0) = local[0];
            }
            break;
        default:
#if defined(__GNUC__)
            __builtin_unreachable();
#endif
    }
}

template <detray::concepts::algebra algebra_t, std::integral size_t, size_t D>
TRACCC_HOST_DEVICE void get_measurement_covariance(
    const measurement& meas, detray::dmatrix<algebra_t, D, D>& cov) {

    static_assert(((D == 1u) || (D == 2u)),
                  "The measurement dimension must be 1 or 2");

    assert((meas.subs.get_indices()[0] == e_bound_loc0) ||
           (meas.subs.get_indices()[0] == e_bound_loc1));

    const variance2& variance = meas.variance;

    switch (meas.subs.get_indices()[0]) {
        case e_bound_loc0:
            getter::element(cov, 0, 0) = variance[0];
            if constexpr (D == 2u) {
                getter::element(cov, 0, 1) = 0.f;
                getter::element(cov, 1, 0) = 0.f;
                getter::element(cov, 1, 1) = variance[1];
            }
            break;
        case e_bound_loc1:
            getter::element(cov, 0, 0) = variance[1];
            if constexpr (D == 2u) {
                getter::element(cov, 0, 1) = 0.f;
                getter::element(cov, 1, 0) = 0.f;
                getter::element(cov, 1, 1) = variance[0];
            }
            break;
        default:
#if defined(__GNUC__)
            __builtin_unreachable();
#endif
    }
}

}  // namespace traccc::edm
