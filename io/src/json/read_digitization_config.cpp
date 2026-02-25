/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_digitization_config.hpp"

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

/// Function allowing the read of @c traccc::module_digitization_config objects
///
/// Note that this function must be declared in the same namespace as
/// @c traccc::module_digitization_config for nlohmann_json to work correctly.
///
traccc::digitization_config read_digitization_config(const nlohmann::json& json) {
    traccc::digitization_config result;

    static const char* entries_key      = "entries";
    static const char* binningdata_key  = "binningdata";
    static const char* bins_key         = "bins";
    static const char* design_index_key = "design_index";
    static const char* id_to_design_key = "id_to_design_map";
    static const char* detray_id_key    = "detray_id";

    // Step 1: read unique designs
    std::map<int, traccc::module_digitization_config> design_map;

    if (json.contains(entries_key)) {
        for (const auto& entry : json[entries_key]) {
            int design_index = entry[design_index_key].get<int>();
            traccc::module_digitization_config cfg;

            if (entry.contains(binningdata_key)) {
                for (const auto& bindata : entry[binningdata_key]) {
                    std::vector<float> centres =
                        bindata[bins_key].get<std::vector<float>>();
                    cfg.bin_edges.push_back(std::move(centres));
                }
                for (const auto& axis : cfg.bin_edges) {
                    if (axis.size() == 1u) {
                        cfg.dimensions = 1;
                        break;
                    }
                }
            }
            design_map[design_index] = std::move(cfg);
        }
    }

    // Step 2: populate result from id_to_design_map
    result.designs.resize(design_map.size());
    for (const auto& [idx, cfg] : design_map) {
        result.designs[idx] = cfg;
    }

    if (json.contains(id_to_design_key)) {
        for (const auto& mapping : json[id_to_design_key]) {
            uint64_t detray_id  = mapping[detray_id_key].get<uint64_t>();
            int      design_idx = mapping[design_index_key].get<int>();
            result.id_to_design_index[detray_id] = design_idx;
        }
    }

    return result;
}

namespace io::json {

digitization_config read_digitization_config(std::string_view filename) {
    // Open the input file. Relying on exceptions for the error handling.
    std::ifstream infile(filename.data(), std::ifstream::binary);
    infile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    // Read the contents of the file into a JSON object.
    nlohmann::json json;
    infile >> json;

    // Construct the object from the JSON configuration.
    return traccc::read_digitization_config(json);
}

}  // namespace io::json
}  // namespace traccc
