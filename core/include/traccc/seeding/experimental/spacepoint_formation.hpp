/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::experimental {

/// Algorithm forming space points out of measurements
///
/// This algorithm performs the local-to-global transformation of the 2D
/// measurements made on every detector module, into 3D spacepoint coordinates.
///
template <typename detector_t>
class spacepoint_formation
    : public algorithm<spacepoint_collection_types::host(
          const detector_t&, const measurement_collection_types::host&)> {

    public:
    /// Constructor for spacepoint_formation
    ///
    /// @param mr is the memory resource
    ///
    spacepoint_formation(vecmem::memory_resource& mr);

    /// Callable operator for the space point formation, based on one single
    /// module
    ///
    /// @param det Detector object
    /// @param measurements A collection of measurements
    /// @return A spacepoint container, with one spacepoint for every
    ///         measurement
    ///
    spacepoint_collection_types::host operator()(
        const detector_t& det,
        const measurement_collection_types::host& measurements) const override;

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc::experimental

#include "traccc/seeding/experimental/spacepoint_formation.ipp"