/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/edm/DeviceMultiTrajectory.hpp"

// System include(s).
#include <stdexcept>
#include <string>

namespace traccc::edm {

DeviceMultiTrajectory::DeviceMultiTrajectory(
    const track_container<default_algebra>::const_view& tracks,
    const Acts::TrackingGeometry& actsGeometry,
    const host_detector& detrayGeometry)
    : m_tracks{tracks},
      m_actsGeometry{actsGeometry},
      m_detrayGeometry(detrayGeometry) {}

auto DeviceMultiTrajectory::size_impl() const -> IndexType {

    return static_cast<IndexType>(m_tracks.states.size());
}

auto DeviceMultiTrajectory::calibratedSize_impl(IndexType istate) const
    -> IndexType {

    // Access the measurement and return its dimensions.
    return static_cast<IndexType>(
        m_tracks.measurements.at(m_tracks.states.measurement_index().at(istate))
            .dimensions());
}

bool DeviceMultiTrajectory::has_impl(Acts::HashedString key, IndexType) const {

    return hasColumn_impl(key);
}

bool DeviceMultiTrajectory::hasColumn_impl(Acts::HashedString key) const {

    // Use the hashing literal from Acts.
    using namespace Acts::HashedStringLiteral;

    // Hard-code the available columns.
    switch (key) {
        case "chi2"_hash:
        case "measdim"_hash:
            return true;
        default:
            return false;
    }
}

std::any DeviceMultiTrajectory::component_impl(Acts::HashedString key,
                                               IndexType istate) const {

    // Use the hashing literal from Acts.
    using namespace Acts::HashedStringLiteral;

    // Extract the requested component.
    switch (key) {
        case "chi2"_hash:
            return &(m_tracks.states.smoothed_chi2().at(istate));
        case "measdim"_hash:
            return &(m_tracks.measurements
                         .at(m_tracks.states.measurement_index().at(istate))
                         .dimensions());
        default:
            throw std::runtime_error(
                "traccc::edm::DeviceMultiTrajectory::component_impl no such "
                "component: " +
                std::to_string(key));
    }
}

auto DeviceMultiTrajectory::parameters_impl(IndexType index) const
    -> ConstTrackStateProxy::ConstParameters {

    // Access the track state's parameters.
    const bound_track_parameters<default_algebra>& deviceParams =
        m_tracks.states.smoothed_params().at(index);

    // Construct the parameters on the fly. Since Acts always expects the
    // parameters to be FP64, and the device code may use FP32.
    Acts::BoundVector actsParams;
    actsParams << deviceParams.bound_local()[0], deviceParams.bound_local()[1],
        deviceParams.phi(), deviceParams.theta(), deviceParams.qop(),
        deviceParams.time();

    // Construct the parameters expected by Acts.
    return ConstTrackStateProxy::ConstParameters{actsParams.data()};
}

auto DeviceMultiTrajectory::covariance_impl(IndexType index) const
    -> ConstTrackStateProxy::ConstCovariance {

    // Access the track state's parameters.
    const bound_track_parameters<default_algebra>& params =
        m_tracks.states.smoothed_params().at(index);

    // Construct the covariance on the fly.
    Acts::BoundSquareMatrix covariance;
    for (unsigned int i = 0; i < Acts::BoundSquareMatrix::RowsAtCompileTime;
         ++i) {
        for (unsigned int j = 0; j < Acts::BoundSquareMatrix::ColsAtCompileTime;
             ++j) {
            covariance(i, j) = params.covariance()[i][j];
        }
    }

    // Construct the covariance expected by Acts.
    return ConstTrackStateProxy::ConstCovariance{covariance.data()};
}

auto DeviceMultiTrajectory::jacobian_impl(IndexType) const
    -> ConstTrackStateProxy::ConstCovariance {

    // Currently not supported in traccc.
    throw std::runtime_error(
        "traccc::edm::DeviceMultiTrajectory::jacobian_impl is not supported");
}

const Acts::Surface* DeviceMultiTrajectory::referenceSurface_impl(
    IndexType index) const {

    // Get the Detray surface barcode belonging to the requested track state.
    const detray::geometry::barcode& barcode =
        m_tracks.measurements.surface_link().at(
            m_tracks.states.measurement_index().at(index));

    // With this barcode, look up the Acts surface using the Detray geometry.
    const Acts::GeometryIdentifier acts_surface_id =
        host_detector_visitor<detector_type_list>(
            m_detrayGeometry.get(),
            [&]<typename detector_traits_t>(
                const typename detector_traits_t::host& detector) {
                return Acts::GeometryIdentifier{
                    detector.surfaces().search(barcode).source};
            });

    // And now that we have an Acts surface identifier, retrieve the surface
    // from the Acts tracking geometry.
    return m_actsGeometry.get().findSurface(acts_surface_id);
}

const std::vector<Acts::HashedString>& DeviceMultiTrajectory::dynamicKeys_impl()
    const {

    // There are no dynamic keys in traccc multi-trajectories.
    static const std::vector<Acts::HashedString> empty_keys{};
    return empty_keys;
}

Acts::SourceLink DeviceMultiTrajectory::getUncalibratedSourceLink_impl(
    IndexType) const {

    return Acts::SourceLink{nullptr};
}

}  // namespace traccc::edm
