/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/io/frontend/writer_interface.hpp"
#include "detray/io/json/json.hpp"
#include "detray/io/json/json_io.hpp"
#include "detray/io/utils/file_handle.hpp"

// System include(s)
#include <cassert>
#include <filesystem>
#include <ios>
#include <stdexcept>
#include <string>

namespace detray::io {

/// @brief Class that adds json functionality to common writer types.
///
/// Assemble the json writers from the common writer types, which serialize a
/// detector into the io payloads, and this class, which does the file
/// handling and provides the json stream. It also inlcudes the respective
/// @c to_json and @c from_json functions for the payloads ("json_serializers").
///
/// @note The resulting writer types will fulfill @c writer_interface through
/// the common writers they are being extended with
template <class detector_t, class writer_backend_t>
class json_writer final : public writer_interface<detector_t> {

    using io_backend = writer_backend_t;

    public:
    /// File gets created with the json file extension
    json_writer() : writer_interface<detector_t>(".json") {}

    /// Writes the geometry to file with a given name
    std::string write(
        const detector_t &det, const typename detector_t::name_map &names,
        const std::ios_base::openmode mode = std::ios::out | std::ios::binary,
        const std::filesystem::path &file_path = {"./"}) override {
        // Assert output stream
        assert(((mode == std::ios_base::out) ||
                (mode == (std::ios_base::out | std::ios_base::binary)) ||
                (mode == (std::ios_base::out | std::ios_base::trunc)) ||
                (mode == (std::ios_base::out | std::ios_base::trunc |
                          std::ios_base::binary))) &&
               "Illegal file mode for json writer");

        // By convention the name of the detector is the first element
        std::string det_name = "";
        if (!names.empty()) {
            det_name = names.at(0);
        }

        // Create a new file
        std::string file_stem{det_name + "_" + std::string(io_backend::tag)};
        io::file_handle file{file_path / file_stem, this->file_extension(),
                             mode};

        // Write some general information
        nlohmann::ordered_json out_json;
        out_json["header"] = io_backend::write_header(det, det_name);

        // Write the detector data into the json stream by using the
        // conversion functions defined in "detray/io/json/json_io.hpp"
        out_json["data"] = io_backend::convert(det, names);

        // Write to file
        *file << std::setw(4) << out_json << std::endl;

        return file_stem + this->file_extension();
    }
};

}  // namespace detray::io
