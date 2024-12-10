/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/builders/detector_builder.hpp"
#include "detray/io/frontend/reader_interface.hpp"
#include "detray/io/json/json.hpp"
#include "detray/io/json/json_io.hpp"
#include "detray/io/utils/file_handle.hpp"

// System include(s)
#include <ios>
#include <iostream>
#include <string>

namespace detray::io {

/// @brief Class that adds json functionality to common reader types.
///
/// Assemble the json readers from the common reader types, which handle the
/// volume builders, and this class, which provides the payload data from the
/// json stream. It also inlcudes the respective @c to_json and @c from_json
/// functions for the payloads ("json_serializers").
///
/// @note The resulting reader types will fulfill @c reader_interface through
/// the common readers they are being extended with
template <class detector_t, class reader_backend_t>
class json_reader final : public reader_interface<detector_t> {

    using io_backend = reader_backend_t;

    public:
    /// Set json file extension
    json_reader() : reader_interface<detector_t>(".json") {}

    /// Writes the geometry to file with a given name
    void read(detector_builder<typename detector_t::metadata, volume_builder>&
                  det_builder,
              typename detector_t::name_map& name_map,
              const std::string& file_name) override {

        // Read json from file
        io::file_handle file{file_name,
                             std::ios_base::in | std::ios_base::binary};

        // Reads the data from file and returns the corresponding io payloads
        nlohmann::json in_json;
        *file >> in_json;

        // Add the data from the payload to the detray detector builder
        io_backend::template convert<detector_t>(det_builder, name_map,
                                                 in_json["data"]);
    }
};

}  // namespace detray::io
