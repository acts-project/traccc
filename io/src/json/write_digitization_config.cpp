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
void to_json(nlohmann::json& json, const module_digitization_config& cfg) {
    static const char* geometric       = "geometric";
    static const char* segmentation    = "segmentation";
    static const char* binningdata_key = "binningdata";

    nlohmann::json binning_array = nlohmann::json::array();

    for (std::size_t dim = 0; dim < cfg.bin_edges.size(); ++dim) {
        nlohmann::json bindata;
        const auto& edges = cfg.bin_edges[dim];

        int nbins = static_cast<int>(edges.size()) - 1;
        bindata["bins"] = nbins;

        if (!edges.empty()) {
            bindata["min"] = edges.front();
            bindata["max"] = edges.back();
        }

        // Detect equidistant vs irregular binning
        bool equidistant = true;
        if (nbins > 1) {
            float expected_pitch = (edges.back() - edges.front()) / static_cast<float>(nbins);
            for (int i = 1; i < static_cast<int>(edges.size()); ++i) {
                if (std::abs((edges[i] - edges[i-1]) - expected_pitch) > 1e-5f) {
                    equidistant = false;
                    break;
                }
            }
        }

        if (equidistant) {
            bindata["type"] = "equidistant";
        } else {
            bindata["type"] = "arbitrary";
            bindata["edges"] = edges;
        }

        binning_array.push_back(bindata);
    }

    json[geometric][segmentation][binningdata_key] = binning_array;
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
