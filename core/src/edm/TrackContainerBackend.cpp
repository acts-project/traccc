/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/edm/TrackContainerBackend.hpp"

// System include(s).
#include <stdexcept>
#include <string>

namespace traccc::edm {

TrackContainerBackend::TrackContainerBackend(
    const track_container<default_algebra>::const_view& tracks,
    const Acts::TrackingGeometry& actsGeometry,
    const host_detector& detrayGeometry)
    : m_tracks{tracks},
      m_actsGeometry{actsGeometry},
      m_detrayGeometry(detrayGeometry) {}

std::size_t TrackContainerBackend::size_impl() const {

    return static_cast<std::size_t>(m_tracks.tracks.size());
}

auto TrackContainerBackend::parameters(IndexType index) const
    -> ConstParameters {

    // Access the track's parameters.
    const bound_track_parameters<default_algebra>& deviceParams =
        m_tracks.tracks.params().at(index);

    // Construct the parameters on the fly. Since Acts always expects the
    // parameters to be FP64, and the device code may use FP32.
    Acts::BoundVector actsParams;
    actsParams << deviceParams.bound_local()[0], deviceParams.bound_local()[1],
        deviceParams.phi(), deviceParams.theta(), deviceParams.qop(),
        deviceParams.time();

    // Construct the parameters expected by Acts.
    return ConstParameters{actsParams.data()};
}

auto TrackContainerBackend::covariance(IndexType index) const
    -> ConstCovariance {

    // Access the track's parameters.
    const bound_track_parameters<default_algebra>& params =
        m_tracks.tracks.params().at(index);

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
    return ConstCovariance{covariance.data()};
}

bool TrackContainerBackend::hasColumn_impl(Acts::HashedString key) const {

    // Use the hashing literal from Acts.
    using namespace Acts::HashedStringLiteral;

    // Hard-code the available columns.
    switch (key) {
        case "fitOutcome"_hash:
        case "params"_hash:
        case "ndf"_hash:
        case "chi2"_hash:
        case "pval"_hash:
        case "nHoles"_hash:
            return true;
        default:
            return false;
    }
}

std::any TrackContainerBackend::component_impl(Acts::HashedString key,
                                               IndexType itrack) const {

    // Use the hashing literal from Acts.
    using namespace Acts::HashedStringLiteral;

    // Hard-code the available columns.
    switch (key) {
        case "fitOutcome"_hash:
            return &(m_tracks.tracks.fit_outcome().at(itrack));
        case "params"_hash:
            return &(m_tracks.tracks.params().at(itrack));
        case "ndf"_hash:
            return &(m_tracks.tracks.ndf().at(itrack));
        case "chi2"_hash:
            return &(m_tracks.tracks.chi2().at(itrack));
        case "pval"_hash:
            return &(m_tracks.tracks.pval().at(itrack));
        case "nHoles"_hash:
            return &(m_tracks.tracks.nholes().at(itrack));
        default:
            throw std::runtime_error(
                "TrackContainerBackend::component_impl: Requested track "
                "component does not exist: " +
                std::to_string(key));
    }
}

Acts::ParticleHypothesis TrackContainerBackend::particleHypothesis_impl(
    IndexType) const {

    // traccc tracks do not have per-track particle hypotheses; return a default
    return Acts::ParticleHypothesis::muon();
}

const Acts::Surface* TrackContainerBackend::referenceSurface_impl(
    IndexType index) const {

    // Get the Detray surface barcode belonging to the first track state of the
    // requested track.
    detray::geometry::barcode barcode;
    if (m_tracks.tracks.at(index).constituent_links().at(0u).type ==
        track_constituent_link::track_state) {
        barcode = m_tracks.measurements.surface_link().at(
            m_tracks.states.measurement_index().at(
                m_tracks.tracks.constituent_links().at(index).at(0u).index));
    } else if (m_tracks.tracks.at(index).constituent_links().at(0u).type ==
               track_constituent_link::measurement) {
        barcode = m_tracks.measurements.surface_link().at(
            m_tracks.tracks.constituent_links().at(index).at(0u).index);
    } else {
        throw std::runtime_error(
            "TrackContainerBackend::referenceSurface_impl: Invalid track "
            "constituent link type encountered");
    }

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

const std::vector<Acts::HashedString>& TrackContainerBackend::dynamicKeys_impl()
    const {

    static const std::vector<Acts::HashedString> emptyKeys{};
    return emptyKeys;
}

}  // namespace traccc::edm
