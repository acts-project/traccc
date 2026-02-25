/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "write_digitization_config.hpp"

// Acts include(s).
#if __has_include(<ActsPlugins/Json/ActsJson.hpp>)
#include <ActsPlugins/Json/ActsJson.hpp>
#include <ActsPlugins/Json/GeometryHierarchyMapJsonConverter.hpp>
#include <ActsPlugins/Json/UtilitiesJsonConverter.hpp>
#else
#include <Acts/Plugins/Json/ActsJson.hpp>
#include <Acts/Plugins/Json/GeometryHierarchyMapJsonConverter.hpp>
#include <Acts/Plugins/Json/UtilitiesJsonConverter.hpp>
#endif

// System include(s).
#include <fstream>

namespace traccc {

/// Function allowing the write of @c traccc::module_digitization_config objects
///
/// Note that this function must be declared in the same namespace as
/// @c traccc::module_digitization_config for nlohmann_json to work correctly.
///
void to_json(nlohmann::json& json, const traccc::digitization_config& cfg) {
    json["entries"] = nlohmann::json::array();

    for (int i = 0; i < (int)cfg.designs.size(); ++i) {
        const auto& design = cfg.designs[i];

        nlohmann::json binningdata = nlohmann::json::array();
        for (const auto& axis : design.bin_edges) {
            binningdata.push_back({{"bins", axis}});
        }

        json["entries"].push_back({
            {"design_index", i},
            {"binningdata", binningdata}
        });
    }

    nlohmann::json id_map = nlohmann::json::array();
    for (const auto& [detray_id, design_idx] : cfg.id_to_design_index) {
        id_map.push_back({
            {"detray_id", detray_id},
            {"design_index", design_idx}
        });
    }
    json["id_to_design_map"] = id_map;
}

namespace io::json {

void write_digitization_config(std::string_view filename,
                               const digitization_config& config) {
    // Construct the JSON object to be written.
    nlohmann::json json;
    to_json(json, config);

    // Open the output file. Relying on exceptions for the error handling.
    std::ofstream outfile(filename.data(), std::ofstream::binary);
    outfile.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    // Write the JSON object to the file.
    outfile << json.dump(4);
}

}  // namespace io::json
}  // namespace traccc
