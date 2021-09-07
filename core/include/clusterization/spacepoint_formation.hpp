/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "utils/algorithm.hpp"

namespace traccc {

/// Connected component labeling.
struct spacepoint_formation
    : public algorithm<host_spacepoint_collection(
          const cell_module&, const host_measurement_collection&)> {

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
    output_type operator()(
        const cell_module& c,
        const host_measurement_collection& m) const override {
        output_type spacepoints;
        this->operator()(c, m, spacepoints);
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
                    output_type& spacepoints) const {
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
