/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/spacepoint_formation.hpp"

namespace traccc {

spacepoint_formation::spacepoint_formation(vecmem::memory_resource& mr)
    : m_mr(mr) {}

spacepoint_formation::output_type spacepoint_formation::operator()(
    const measurement_container_types::host& measurements) const {

    // Create the result container, with the correct "outer size".
    output_type result(measurements.size(), &(m_mr.get()));

    // Iterate over the modules.
    for (std::size_t i = 0; i < measurements.size(); ++i) {

        // Access the measurements of the current module.
        const cell_module& module = measurements.get_headers()[i];
        const measurement_collection_types::host& measurements_per_module =
            measurements.get_items()[i];

        // Set the geometry ID for this collection of spacepoints.
        result[i].header = module.module;

        // Access the spacepoint collection for the module.
        spacepoint_collection_types::host& spacepoints_per_module =
            result[i].items;
        spacepoints_per_module.reserve(measurements_per_module.size());

        // Construct the spacepoints.
        for (const measurement& m : measurements_per_module) {

            point3 local_3d = {m.local[0], m.local[1], 0.};
            point3 global = module.placement.point_to_global(local_3d);
            spacepoints_per_module.push_back({global, m});
        }
    }

    // Return the created container.
    return result;
}

}  // namespace traccc
