/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_digitization_config.hpp"

// Acts include(s).
#include <Acts/Plugins/Json/ActsJson.hpp>
#include <Acts/Plugins/Json/GeometryHierarchyMapJsonConverter.hpp>
#include <Acts/Plugins/Json/UtilitiesJsonConverter.hpp>

// System include(s).
#include <fstream>

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
    static const char* binningdata = "binningdata";
    static const char* bins = "bins";

    // Read the binning information, if possible.
    if (json.find(geometric) != json.end()) {
        const auto& json_geom = json[geometric];
        if (json_geom.find(segmentation) != json_geom.end()) {
            Acts::from_json(json_geom[segmentation], cfg.segmentation);
            // If we only have 1 bins along any axis, then this is a 1D module.
            const auto& json_segm = json_geom[segmentation];
            for (const auto& bindata : json_segm[binningdata]) {
                if (bindata[bins].get<int>() == 1) {
                    cfg.dimensions = 1;
                    break;
                }
            }
        }
    }
}

namespace io::json {

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

}  // namespace io::json
}  // namespace traccc
