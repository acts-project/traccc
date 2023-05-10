/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_measurements.hpp"

#include "make_measurement_reader.hpp"

// System include(s).
#include <algorithm>

namespace traccc::io::csv {

void read_measurements(measurement_reader_output& out,
                       std::string_view filename) {

    // Construct the measurement reader object.
    auto reader = make_measurement_reader(filename);

    // Create the result collection.
    alt_measurement_collection_types::host& result_measurements =
        out.measurements;
    cell_module_collection_types::host& result_modules = out.modules;

    std::map<geometry_id, unsigned int> m;

    // Read the measurements from the input file.
    csv::measurement iomeas;
    while (reader.read(iomeas)) {

        unsigned int link;
        auto it = m.find(iomeas.geometry_id);

        if (it != m.end()) {
            link = (*it).second;
        } else {
            link = result_modules.size();
            m[iomeas.geometry_id] = link;
            cell_module mod;
            mod.module = iomeas.geometry_id;
            result_modules.push_back(mod);
        }

        // Construct the measurement object.
        const traccc::alt_measurement meas{
            point2{iomeas.local0, iomeas.local1},
            variance2{iomeas.var_local0, iomeas.var_local1}, link};

        result_measurements.push_back(meas);
    }
}

}  // namespace traccc::io::csv
