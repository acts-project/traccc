/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/write_digitization_config.hpp"

#include "traccc/io/utils.hpp"

// Acts include(s).
#include <Acts/Plugins/Json/ActsJson.hpp>
#include <Acts/Plugins/Json/UtilitiesJsonConverter.hpp>

// System include(s).
#include <fstream>
#include <string>

namespace traccc {

/// Function allowing the writing of @c traccc::module_digitization_config
/// objects
inline void to_json(nlohmann::json& json,
                    const module_digitization_config& cfg) {

    json["geometric"]["indices"] = cfg.indices;
    json["geometric"]["segmentation"] = cfg.segmentation;
}

/// Function allowing the writing of digitization config objects from a
/// collection of detray barcodes and @c traccc::module_digitization_config
inline void to_json(nlohmann::json& in_json,
                    const io::digitization_out_collection& cfg) {

    nlohmann::json entries = nlohmann::json::array();
    nlohmann::json entry = nlohmann::json::object();
    for (const auto& c : cfg) {
        entry["volume"] = c.first.volume();
        entry["sensitive"] = c.first.index();
        if (c.first.extra() != 255) {
            entry["extra"] = c.first.extra();
        }
        entry["value"] = c.second;
        entries.push_back(entry);
    }
    in_json = entries;
}

namespace io {
namespace json {

void write_digitization_config(std::string_view filename,
                               const digitization_out_collection& digi_cfg) {

    // Open the output file. Relying on exceptions for the error handling.
    std::ofstream outfile(filename.data(), std::ifstream::binary);
    outfile.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    // Write the digitization data into the json stream
    nlohmann::json json;

    // Write header
    json["acts-geometry-hierarchy-map"]["format-version"] = 0;
    json["acts-geometry-hierarchy-map"]["value-identifier"] =
        "digitization-configuration";

    // Write entries
    json["entries"] = digi_cfg;

    // Write to file
    outfile << std::setw(2) << json << std::endl;

    if (outfile.bad()) {
        std::cout << "ERROR: Could not write to file";
    }
    outfile.close();
}

}  // namespace json

void write_digitization_config(std::string_view filename,
                               const digitization_out_collection& digi_cfg,
                               data_format format) {

    // Construct the full filename.
    std::string full_filename = filename.data();

    // Decide how to write the file.
    switch (format) {
        case data_format::json:
            return json::write_digitization_config(full_filename, digi_cfg);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace io
}  // namespace traccc
