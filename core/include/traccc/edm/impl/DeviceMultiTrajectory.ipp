/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::edm {

template <std::size_t MEASDIM>
auto DeviceMultiTrajectory::calibrated_impl(IndexType istate) const
    -> ConstTrackStateProxy::Calibrated<MEASDIM> {

    return ConstTrackStateProxy::Calibrated<MEASDIM>{
        m_tracks.measurements.at(m_tracks.states.measurement_index().at(istate))
            .local_position()
            .data()};
}

template <std::size_t MEASDIM>
auto DeviceMultiTrajectory::calibratedCovariance_impl(IndexType) const
    -> ConstTrackStateProxy::CalibratedCovariance<MEASDIM> {

    // We don't provide this at the moment.
    throw std::runtime_error(
        "traccc::edm::DeviceMultiTrajectory::calibratedCovariance_impl is not "
        "supported");
}

}  // namespace traccc::edm
