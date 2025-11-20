/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/edm/track_container.hpp"
#include "traccc/geometry/host_detector.hpp"

// Acts include(s).
#include <Acts/EventData/MultiTrajectory.hpp>
#include <Acts/EventData/TrackContainer.hpp>
#include <Acts/EventData/TrackContainerBackendConcept.hpp>
#include <Acts/EventData/TrackStateProxy.hpp>
#include <Acts/Geometry/TrackingGeometry.hpp>

// System include(s).
#include <any>
#include <functional>

namespace traccc::edm {

/// Implementation of the @c Acts::ConstTrackContainerBackend concept for traccc
/// produced tracks
class DeviceTrackBackend {

    public:
    /// Constructor from a track container and Acts/Detray tracking geometries
    ///
    /// @param tracks The track container holding the data
    /// @param actsGeometry The Acts tracking geometry associated with the
    ///                     trajectories
    /// @param detrayGeometry The Detray tracking geometry associated with the
    ///                       trajectories
    ///
    explicit DeviceTrackBackend(
        const track_container<default_algebra>::const_view& tracks,
        const Acts::TrackingGeometry& actsGeometry,
        const host_detector& detrayGeometry);
    /// Copy constructor
    DeviceTrackBackend(const DeviceTrackBackend&) = default;

    /// Copy assignment operator
    DeviceTrackBackend& operator=(const DeviceTrackBackend&) = default;

    /// Track index type
    using IndexType = Acts::MultiTrajectoryTraits::IndexType;
    /// Track parameters type
    using ConstParameters =
        Acts::detail_lt::FixedSizeTypes<Acts::eBoundSize,
                                        true>::CoefficientsMap;
    /// Track covariance type
    using ConstCovariance =
        Acts::detail_lt::FixedSizeTypes<Acts::eBoundSize, true>::CovarianceMap;

    /// @name Functions required by @c Acts::TrackContainer
    /// @{

    /// Get the size of the object (the number of tracks)
    ///
    /// @return The total number of tracks
    ///
    std::size_t size_impl() const;

    /// Retrieve the parameters of a given track
    ///
    /// @param index Index into the parameter column
    /// @return The Eigen object representing the parameters
    ///
    ConstParameters parameters(IndexType index) const;

    /// Retrieve the covariance of a given track state
    ///
    /// @param index Index into the covariance column
    /// @return The Eigen object representing the covariance
    ///
    ConstCovariance covariance(IndexType index) const;

    /// Check if a given track column exists
    ///
    /// @param key The key of the column to check
    /// @return @c true if the column exists, @c false otherwise
    ///
    bool hasColumn_impl(Acts::HashedString key) const;

    /// Retrieve a given component of a track
    ///
    /// @param key The key of the component to retrieve
    /// @param itrack The index of the track to retrieve from
    /// @return The component as an @c std::any object
    ///
    std::any component_impl(Acts::HashedString key, IndexType itrack) const;

    /// Particle hypothesis of a given track
    ///
    /// @param itrack The index of the track
    /// @return The particle hypothesis
    ///
    Acts::ParticleHypothesis particleHypothesis_impl(IndexType itrack) const;

    /// Retrieve the reference surface of a given track
    ///
    /// @param index Index into the track container
    /// @return A pointer to the reference surface
    ///
    const Acts::Surface* referenceSurface_impl(IndexType index) const;

    /// Dynamic keys available in this track container
    ///
    /// By definition, it will always be empty for traccc tracks.
    ///
    /// @return The list of dynamic keys
    ///
    const std::vector<Acts::HashedString>& dynamicKeys_impl() const;

    /// @}

    private:
    /// The track container holding the data
    track_container<default_algebra>::const_device m_tracks;
    /// The tracking geometry associated with the trajectories
    std::reference_wrapper<const Acts::TrackingGeometry> m_actsGeometry;
    /// The Detray tracking geometry associated with the trajectories
    std::reference_wrapper<const host_detector> m_detrayGeometry;

};  // class DeviceTrackBackend

/// Make sure that @c DeviceTrackBackend meets the concept requirements
static_assert(Acts::ConstTrackContainerBackend<DeviceTrackBackend>);

}  // namespace traccc::edm

namespace Acts {

/// Declare that @c traccc::edm::DeviceTrackBackend is a read-only track
/// container
template <>
struct IsReadOnlyTrackContainer<traccc::edm::DeviceTrackBackend>
    : std::true_type {};

}  // namespace Acts
