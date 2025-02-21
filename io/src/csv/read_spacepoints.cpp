/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_spacepoints.hpp"

#include "read_measurements.hpp"
#include "traccc/io/csv/make_hit_reader.hpp"
#include "traccc/io/csv/make_measurement_hit_id_reader.hpp"

// System include(s).
#include <algorithm>
#include <stdexcept>

namespace traccc::io::csv {

void read_spacepoints(edm::spacepoint_collection::host& spacepoints,
                      measurement_collection_types::host& measurements,
                      std::string_view hit_filename,
                      std::string_view meas_filename,
                      std::string_view meas_hit_map_filename,
                      const traccc::default_detector::host* detector) {

    // Read all measurements.
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

        // Find the index of the measurement that this hit/spacepoint belongs
        // to. Which may not be valid, as some simulated hits are not associated
        // with a measurement.
        auto const measurement_id_it =
            std::find_if(measurement_hit_ids.begin(), measurement_hit_ids.end(),
                         [&](const measurement_hit_id& mh_id) {
                             return mh_id.hit_id == spacepoints.size();
                         });
        const unsigned int measurement_index =
            (measurement_id_it != measurement_hit_ids.end())
                ? static_cast<unsigned int>(measurement_id_it->measurement_id)
                : static_cast<unsigned int>(-1);

        // Create a new spacepoint for the SoA container.
        spacepoints.push_back(
            {measurement_index, {iohit.tx, iohit.ty, iohit.tz}, 0.f, 0.f});
    }
}

}  // namespace traccc::io::csv
