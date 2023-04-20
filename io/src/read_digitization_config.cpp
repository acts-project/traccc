/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_digitization_config.hpp"

#include "traccc/io/utils.hpp"

// Acts include(s).
#include <Acts/Plugins/Json/ActsJson.hpp>
#include <Acts/Plugins/Json/GeometryHierarchyMapJsonConverter.hpp>
#include <Acts/Plugins/Json/UtilitiesJsonConverter.hpp>

// System include(s).
#include <fstream>
#include <string>

namespace traccc {

/// Function allowing the read of @c traccc::module_digitization_config objects
///
/// Note that this function must be declared in the same namespace as
/// @c traccc::module_digitization_config for nlohmann_json to work correctly.
///
void from_json(const nlohmann::json& json, module_digitization_config& cfg) {

    // Names/keywords used in the JSON file.
    static const char* geometric = "geometric";
    static const char* segmentation = "segmentation";

    // Read the object, if possible.
    if (json.find(geometric) != json.end()) {
        from_json(json[geometric][segmentation], cfg.segmentation);
    }
}

namespace io {
namespace json {

digitization_config read_digitization_config(std::string_view filename) {

    // Open the input file. Relying on exceptions for the error handling.
    std::ifstream infile(filename.data(), std::ifstream::binary);
    infile.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    // Read the contents of the file into a JSON object.
    nlohmann::json json;
    infile >> json;

    // Construct the object from the JSON configuration.
    static const Acts::GeometryHierarchyMapJsonConverter<
        module_digitization_config>
        converter{"digitization-configuration"};
    return converter.fromJson(json);
}

}  // namespace json

digitization_config read_digitization_config(std::string_view filename,
                                             data_format format) {

    // Construct the full filename.
    std::string full_filename = data_directory() + filename.data();

    // Decide how to read the file.
    switch (format) {
        case data_format::json:
            return json::read_digitization_config(full_filename);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace io
}  // namespace traccc
