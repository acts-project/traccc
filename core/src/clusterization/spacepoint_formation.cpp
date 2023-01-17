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
    const alt_measurement_collection_types::host& measurements,
    const cell_module_collection_types::host& modules) const {

    // Create the result container.
    output_type result(&(m_mr.get()));

    geometry_id last_module = 0;

    // Iterate over the measurements.
    for (std::size_t i = 0; i < measurements.size(); ++i) {

        // Access the measurements of the current module.
        const alt_measurement& this_measurement = measurements.at(i);
        const cell_module& module = modules.at(this_measurement.module_link);

        // Transform measurement position to 3D
        point3 local_3d = {this_measurement.local[0], this_measurement.local[1], 0.};
        point3 global = module.placement.point_to_global(local_3d);

        // If new module, create new bin in result jagged vector.
        // TODO: This will be changed soon to alt_spacepoints without jaggedness
        // The headers are not really needed as they are not being used
        // downstream. Safe to admit here that measurements are sorted by module
        if (module.module != last_module) {
            last_module = module.module;
            result.push_back(geometry_id(last_module),
                             spacepoint_collection_types::host(&(m_mr.get())));
        }
        // Fill result with this spacepoint
        result.get_items().back().push_back({global, {this_measurement.local, this_measurement.variance}});
    }

    // Return the created container.
    return result;
}

}  // namespace traccc
