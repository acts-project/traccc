/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "detray/io/frontend/payloads.hpp"
#include "detray/io/json/json.hpp"

/// @brief  The detray JSON I/O is written in such a way that it
/// can read/write ACTS files that are written with the Detray
/// JSON I/O extension
namespace detray::io {

inline void to_json(nlohmann::ordered_json& j, const transform_payload& t) {
    j["translation"] = t.tr;
    j["rotation"] = t.rot;
}

inline void from_json(const nlohmann::ordered_json& j, transform_payload& t) {
    t.tr = j["translation"];
    t.rot = j["rotation"];
}

}  // namespace detray::io
