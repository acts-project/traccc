/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"

namespace traccc {

/// Connected component labeling.
struct spacepoint_formation {

    /// Callable operator for the space point formation, based on one single
    /// module
    ///
    /// @param measurements are the input measurements, in this pixel
    /// demonstrator it one space
    ///    point per measurement
    ///
    /// C++20 piping interface
    ///
    /// @return a measurement collection - size of input/output container is
    /// identical
    host_spacepoint_collection operator()(
        const cell_module& module,
        const host_measurement_collection& measurements) const {

        host_spacepoint_collection spacepoints;
        this->operator()(module, measurements, spacepoints);
        return spacepoints;
    }

    /// Callable operator for the space point formation, based on one single
    /// module
    ///
    /// @param measurements are the input measurements, in this pixel
    /// demonstrator it one space
    ///    point per measurement
    ///
    /// void interface
    ///
    /// @return a measurement collection - size of input/output container is
    /// identical
    void operator()(const cell_module& module,
                    const host_measurement_collection& measurements,
                    host_spacepoint_collection& spacepoints) const {
        // Run the algorithm
        spacepoints.reserve(measurements.size());
        for (const auto& m : measurements) {
            spacepoint s;
            point3 local_3d = {m.local[0], m.local[1], 0.};
            s.global = module.placement.point_to_global(local_3d);
            // @todo add variance estimation
            spacepoints.push_back(std::move(s));
        }
    }
};

}  // namespace traccc
