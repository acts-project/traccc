/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_spacepoints.hpp"

#include "read_measurements.hpp"
#include "traccc/io/csv/make_hit_reader.hpp"
#include "traccc/io/csv/make_measurement_hit_id_reader.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"

// System include(s).
#include <algorithm>
#include <map>

namespace traccc::io::csv {

void read_spacepoints(spacepoint_reader_output& out, std::string_view filename,
                      std::string_view meas_filename,
                      std::string_view meas_hit_map_filename,
                      const geometry& geom) {
    // Read measurements
    measurement_reader_output meas_reader_out;
    read_measurements(meas_reader_out, meas_filename, false);

    // Measurement hit id reader
    auto mhid_reader =
        io::csv::make_measurement_hit_id_reader(meas_hit_map_filename);
    std::vector<traccc::io::csv::measurement_hit_id> measurement_hit_ids;
    traccc::io::csv::measurement_hit_id io_mh_id;
    while (mhid_reader.read(io_mh_id)) {
        measurement_hit_ids.push_back(io_mh_id);
    }

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
            mod.surface_link = detray::geometry::barcode{iohit.geometry_id};
            mod.placement = geom.at(iohit.geometry_id);
            result_modules.push_back(mod);
        }

        // Construct the global 3D position of the spacepoint.
        const point3 pos{iohit.tx, iohit.ty, iohit.tz};

        // Construct the local 3D(2D) position of the measurement.
        measurement meas;
        for (auto const [meas_id, hit_id] : measurement_hit_ids) {
            if (hit_id == result_spacepoints.size()) {
                meas = meas_reader_out.measurements[meas_id];
            }
        }

        // Create the spacepoint object (with its member measurement) from all
        // this information.
        const traccc::spacepoint sp{pos, meas};

        result_spacepoints.push_back(sp);
    }
}

}  // namespace traccc::io::csv
