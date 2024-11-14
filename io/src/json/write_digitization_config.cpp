/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "write_digitization_config.hpp"

// Acts include(s).
#include <Acts/Plugins/Json/ActsJson.hpp>
#include <Acts/Plugins/Json/GeometryHierarchyMapJsonConverter.hpp>
#include <Acts/Plugins/Json/UtilitiesJsonConverter.hpp>

// System include(s).
#include <fstream>

namespace traccc {

/// Function allowing the write of @c traccc::module_digitization_config objects
///
/// Note that this function must be declared in the same namespace as
/// @c traccc::module_digitization_config for nlohmann_json to work correctly.
///
void to_json(nlohmann::json& json, const module_digitization_config& cfg) {

    // Names/keywords used in the JSON file.
    static const char* geometric = "geometric";
    static const char* segmentation = "segmentation";

    // Write the binning information.
    json[geometric][segmentation] = cfg.segmentation;

    // The dimensions variable is determined on reading from the segmentation
    // information, so it does not need to be written separately.
}

namespace io::json {

void write_digitization_config(std::string_view filename,
                               const digitization_config& config) {

    // Construct the JSON object to be written.
    static const Acts::GeometryHierarchyMapJsonConverter<
        module_digitization_config>
        converter{"digitization-configuration"};
    const nlohmann::json json = converter.toJson(config, nullptr);

    // Open the input file. Relying on exceptions for the error handling.
    std::ofstream outfile(filename.data(), std::ifstream::binary);
    outfile.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    // Write the JSON object to the file.
    outfile << json.dump(4);
}

}  // namespace io::json
}  // namespace traccc
