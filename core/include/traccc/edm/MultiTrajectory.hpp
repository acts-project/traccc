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
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/MultiTrajectoryBackendConcept.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"

// System include(s).
#include <any>
#include <functional>

namespace traccc::edm {

/// @c Acts::MultiTrajectory specialisation for traccc produced tracks
class MultiTrajectory : public Acts::MultiTrajectory<MultiTrajectory> {

    public:
    /// Constructor from a track container and Acts/Detray tracking geometries
    ///
    /// @param tracks The track container holding the data
    /// @param actsGeometry The Acts tracking geometry associated with the
    ///                     trajectories
    /// @param detrayGeometry The Detray tracking geometry associated with the
    ///                       trajectories
    ///
    explicit MultiTrajectory(
        const track_container<default_algebra>::const_view& tracks,
        const Acts::TrackingGeometry& actsGeometry,
        const host_detector& detrayGeometry);
    /// Copy constructor
    MultiTrajectory(const MultiTrajectory&) = default;

    /// Copy assignment operator
    MultiTrajectory& operator=(const MultiTrajectory&) = default;

    /// @name Functions required by @c Acts::MultiTrajectory
    /// @{

    /// Get the size of the object (number of track states)
    ///
    /// @return The total number of track states
    ///
    IndexType size_impl() const;

    /// Get the calibrated size of a given track state (measurement dimension)
    ///
    /// @param istate The index of the track state
    /// @return The calibrated size (measurement dimension)
    ///
    IndexType calibratedSize_impl(IndexType istate) const;

    /// Check if a given component exists for a given track state
    ///
    /// @param key The key of the component to check
    /// @param istate The index of the track state to check
    /// @return @c true if the component exists, @c false otherwise
    ///
    bool has_impl(Acts::HashedString key, IndexType istate) const;

    /// Check if a given track state column exists
    ///
    /// @param key The key of the column to check
    /// @return @c true if the column exists, @c false otherwise
    ///
    bool hasColumn_impl(Acts::HashedString key) const;

    /// Retrieve a given component of a track state
    ///
    /// @param key The key of the component to retrieve
    /// @param istate The index of the track state to retrieve from
    /// @return The component as an @c std::any object
    ///
    std::any component_impl(Acts::HashedString key, IndexType istate) const;

    /// Retrieve the parameters of a given track state
    ///
    /// @param index Index into the parameter column
    /// @return The Eigen object representing the parameters
    ///
    ConstTrackStateProxy::ConstParameters parameters_impl(
        IndexType index) const;

    /// Retrieve the covariance of a given track state
    ///
    /// @param index Index into the covariance column
    /// @return The Eigen object representing the covariance
    ///
    ConstTrackStateProxy::ConstCovariance covariance_impl(
        IndexType index) const;

    /// Retrieve the calibrated measurement (positions) of a given track state
    ///
    /// @tparam MEASDIM the measurement dimension
    /// @param istate The track state index
    /// @return The Eigen object representing the calibrated measurement
    ///
    template <std::size_t MEASDIM>
    ConstTrackStateProxy::Calibrated<MEASDIM> calibrated_impl(
        IndexType istate) const;

    /// Retrieve the calibrated measurement covariance of a given track state
    ///
    /// @tparam MEASDIM the measurement dimension
    /// @param index The track state index
    /// @return The Eigen object representing the calibrated measurement
    ///         covariance
    ///
    template <std::size_t MEASDIM>
    ConstTrackStateProxy::CalibratedCovariance<MEASDIM>
    calibratedCovariance_impl(IndexType index) const;

    /// Retrieve the jacobian of a given track state
    ///
    /// @param istate The track state
    /// @return The Eigen object representing the jacobian
    ///
    ConstTrackStateProxy::ConstCovariance jacobian_impl(IndexType istate) const;

    /// Retrieve the reference surface of a given track state
    ///
    /// @param index Index into the track state container
    /// @return A pointer to the reference surface
    ///
    const Acts::Surface* referenceSurface_impl(IndexType index) const;

    /// Dynamic keys available in this multi-trajectory
    ///
    /// By definition, it will always be empty for traccc multi-trajectories.
    ///
    /// @return The list of dynamic keys
    ///
    const std::vector<Acts::HashedString>& dynamicKeys_impl() const;

    /// Retrieve the uncalibrated source link of a given track state
    ///
    /// @param istate The index of the track state
    /// @return The source link object
    ///
    Acts::SourceLink getUncalibratedSourceLink_impl(IndexType istate) const;

    /// @}

    private:
    /// The track container holding the data
    track_container<default_algebra>::const_device m_tracks;
    /// The tracking geometry associated with the trajectories
    std::reference_wrapper<const Acts::TrackingGeometry> m_actsGeometry;
    /// The Detray tracking geometry associated with the trajectories
    std::reference_wrapper<const host_detector> m_detrayGeometry;

};  // class DeviceMultiTrajectory

/// Make sure that @c MultiTrajectory meets the concept requirements
static_assert(Acts::ConstMultiTrajectoryBackend<MultiTrajectory>);

}  // namespace traccc::edm

namespace Acts {

/// Declare that @c traccc::edm::MultiTrajectory is a read-only multi-trajectory
template <>
struct IsReadOnlyMultiTrajectory<traccc::edm::MultiTrajectory>
    : std::true_type {};

}  // namespace Acts

// Include the template implementation.
#include "traccc/edm/impl/MultiTrajectory.ipp"
