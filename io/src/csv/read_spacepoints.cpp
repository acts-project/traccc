/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_spacepoints.hpp"

#include "make_hit_reader.hpp"

// System include(s).
#include <algorithm>
#include <map>

namespace traccc::io::csv {

void read_spacepoints(spacepoint_reader_output& out, std::string_view filename,
                      const geometry& geom) {

    // Construct the spacepoint reader object.
    auto reader = make_hit_reader(filename);

    // Create the result collection.
    spacepoint_collection_types::host& result_spacepoints = out.spacepoints;
    cell_module_collection_types::host& result_modules = out.modules;

    std::map<geometry_id, unsigned int> m;

    // Read the spacepoints from the input file.
    hit iohit;
    while (reader.read(iohit)) {
        unsigned int link;
        auto it = m.find(iohit.geometry_id);
        if (it != m.end()) {
            link = (*it).second;
        } else {
            link = result_modules.size();
            m[iohit.geometry_id] = link;
            cell_module mod;
            mod.module = iohit.geometry_id;
            mod.placement = geom[iohit.geometry_id];
            result_modules.push_back(mod);
        }

        // Find the local<->global transformation for the spacepoint's detector
        // module.
        const transform3& placement = geom[iohit.geometry_id];

        // Construct the global 3D position of the spacepoint.
        const point3 pos{iohit.tx, iohit.ty, iohit.tz};

        // Construct the local 3D(2D) position of the measurement.
        const point3 lpos = placement.point_to_local(pos);

        // Create the spacepoint object (with its member measurement) from all
        // this information.
        const traccc::spacepoint sp{
            pos, {point2{lpos[0], lpos[1]}, variance2{0., 0.}, link}};

        result_spacepoints.push_back(sp);
    }
}

}  // namespace traccc::io::csv
