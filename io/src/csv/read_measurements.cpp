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

measurement_container_types::host read_measurements_container(
    std::string_view filename, vecmem::memory_resource* mr) {

    // Construct the measurement reader object.
    auto reader = make_measurement_reader(filename);

    // Create the result container.
    measurement_container_types::host result;
    if (mr != nullptr) {
        result = measurement_container_types::host{mr};
    }

    // Read the measurements from the input file.
    csv::measurement iomeas;
    while (reader.read(iomeas)) {

        // Construct the module ID for the measurement.
        cell_module module;
        module.module = iomeas.geometry_id;

        // Construct the measurement object.
        const traccc::measurement meas{
            point2{iomeas.local0, iomeas.local1},
            variance2{iomeas.var_local0, iomeas.var_local1}};

        // Find the detector module that this measurement belongs to.
        const measurement_container_types::host::header_vector& headers =
            result.get_headers();
        auto rit = std::find(headers.rbegin(), headers.rend(), module);

        // Add the measurement to the correct place in the container.
        if (rit == headers.rend()) {
            if (mr != nullptr) {
                result.push_back(
                    module,
                    measurement_container_types::host::item_vector::value_type(
                        {meas}, mr));
            } else {
                result.push_back(
                    module,
                    measurement_container_types::host::item_vector::value_type(
                        {meas}));
            }
        } else {
            // The reverse iterator.base() returns the equivalent normal
            // iterator shifted by 1, so that the (r)end and (r)begin iterators
            // match consistently, due to the extra past-the-last element
            auto idx = std::distance(headers.begin(), rit.base()) - 1;
            result.at(idx).items.push_back(meas);
        }
    }

    // Return the container.
    return result;
}

}  // namespace traccc::io::csv
