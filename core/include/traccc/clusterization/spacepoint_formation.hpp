/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc {

/// Connected component labeling.
struct spacepoint_formation
    : public algorithm<host_spacepoint_collection(
          const cell_module&, const host_measurement_collection&)> {

    public:
    /// Constructor for spacepoint_formation
    ///
    /// @param mr is the memory resource
    spacepoint_formation(vecmem::memory_resource& mr) : m_mr(mr) {}

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
            point3 local_3d = {m.local[0], m.local[1], 0.};
            point3 global = module.placement.point_to_global(local_3d);
            spacepoint s({global, m});

            spacepoints.push_back(std::move(s));
        }
    }

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc
