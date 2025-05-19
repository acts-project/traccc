/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::edm {

template <typename BASE>
TRACCC_HOST_DEVICE bool track_state<BASE>::is_hole() const {

    return (state() & IS_HOLE_MASK);
}

template <typename BASE>
TRACCC_HOST_DEVICE void track_state<BASE>::set_hole(bool value) {

    if (value) {
        state() |= IS_HOLE_MASK;
    } else {
        state() &= ~IS_HOLE_MASK;
    }
}

template <typename BASE>
TRACCC_HOST_DEVICE bool track_state<BASE>::is_smoothed() const {

    return (state() & IS_SMOOTHED_MASK);
}

template <typename BASE>
TRACCC_HOST_DEVICE void track_state<BASE>::set_smoothed(bool value) {

    if (value) {
        state() |= IS_SMOOTHED_MASK;
    } else {
        state() &= ~IS_SMOOTHED_MASK;
    }
}

template <typename BASE>
template <detray::concepts::algebra ALGEBRA, std::integral size_type,
          size_type D>
TRACCC_HOST_DEVICE void track_state<BASE>::get_measurement_local(
    const measurement_collection_types::const_device& measurements,
    detray::dmatrix<ALGEBRA, D, 1>& pos) const {

    static_assert(((D == 1u) || (D == 2u)),
                  "The measurement dimension must be 1 or 2");

    assert((measurements.at(measurement_index()).subs.get_indices()[0] ==
            e_bound_loc0) ||
           (measurements.at(measurement_index()).subs.get_indices()[0] ==
            e_bound_loc1));

    const point2& local = measurements.at(measurement_index()).local;

    switch (measurements.at(measurement_index()).subs.get_indices()[0]) {
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

template <typename BASE>
template <detray::concepts::algebra ALGEBRA, std::integral size_type,
          size_type D>
TRACCC_HOST_DEVICE void track_state<BASE>::get_measurement_covariance(
    const measurement_collection_types::const_device& measurements,
    detray::dmatrix<ALGEBRA, D, D>& cov) const {

    static_assert(((D == 1u) || (D == 2u)),
                  "The measurement dimension must be 1 or 2");

    assert((measurements.at(measurement_index()).subs.get_indices()[0] ==
            e_bound_loc0) ||
           (measurements.at(measurement_index()).subs.get_indices()[0] ==
            e_bound_loc1));

    const variance2& variance = measurements.at(measurement_index()).variance;

    switch (measurements.at(measurement_index()).subs.get_indices()[0]) {
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
