/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/detector_description.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::host {

/// Algorithm forming space points out of measurements
///
/// This algorithm performs the local-to-global transformation of the 2D
/// measurements made on every detector module, into 3D spacepoint coordinates.
///
class spacepoint_formation_algorithm
    : public algorithm<spacepoint_collection_types::host(
          const measurement_collection_types::const_view&,
          const detector_description::const_view&)> {

    public:
    /// Constructor for spacepoint_formation
    ///
    /// @param mr is the memory resource
    ///
    spacepoint_formation_algorithm(vecmem::memory_resource& mr);

    /// Callable operator for the space point formation, based on one single
    /// module
    ///
    /// @param measurements_view A collection of measurements
    /// @param dd_view           The detector description
    /// @return The reconstructed spacepoint collection
    ///
    output_type operator()(
        const measurement_collection_types::const_view& measurements_view,
        const detector_description::const_view& dd_view) const override;

    private:
    /// The memory resource used by the algorithm
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class spacepoint_formation_algorithm

}  // namespace traccc::host
