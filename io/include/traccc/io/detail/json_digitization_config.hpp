/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts include(s)
#include "Acts/Plugins/Json/ActsJson.hpp"
#include "Acts/Plugins/Json/UtilitiesJsonConverter.hpp"
#include "Acts/Utilities/BinUtility.hpp"

namespace traccc {

struct digitization_config {
    Acts::BinUtility segmentation;
};

inline void from_json(const nlohmann::json& j, digitization_config& geo_cfg) {
    if (j.find("geometric") != j.end()) {
        nlohmann::json jgdc = j["geometric"];
        from_json(jgdc["segmentation"], geo_cfg.segmentation);
    }
}

}  // namespace traccc