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

void read_spacepoints(spacepoint_collection_types::host& spacepoints,
                      std::string_view hit_filename,
                      std::string_view meas_filename,
                      std::string_view meas_hit_map_filename,
                      const traccc::default_detector::host* detector) {

    // Read all measurements.
    measurement_collection_types::host measurements;
    static constexpr bool sort_measurements = false;
    read_measurements(measurements, meas_filename, detector, sort_measurements);

    // Measurement hit id reader
    auto mhid_reader =
        io::csv::make_measurement_hit_id_reader(meas_hit_map_filename);
    std::vector<traccc::io::csv::measurement_hit_id> measurement_hit_ids;
    traccc::io::csv::measurement_hit_id io_mh_id;
    while (mhid_reader.read(io_mh_id)) {
        measurement_hit_ids.push_back(io_mh_id);
    }

    // Construct the hit reader object.
    auto hit_reader = make_hit_reader(hit_filename);

    // Read the hits from the input file.
    hit iohit;
    while (hit_reader.read(iohit)) {

        // Construct the global 3D position of the spacepoint.
        const point3 pos{iohit.tx, iohit.ty, iohit.tz};

        // Construct the local 3D(2D) position of the measurement.
        measurement meas;
        for (auto const [meas_id, hit_id] : measurement_hit_ids) {
            if (hit_id == spacepoints.size()) {
                meas = measurements[meas_id];
            }
        }

        // Create the spacepoint object (with its member measurement) from all
        // this information.
        spacepoints.push_back({pos, meas});
    }
}

}  // namespace traccc::io::csv
