/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace traccc {

/// Type describing the digitization configuration of a detector module
struct module_digitization_config {
    std::vector<std::vector<float>> bin_edges; // one vector per axis (X, Y)
    unsigned char dimensions = 2;
};

/// Type describing the digitization configuration for the whole detector
struct digitization_config {
    /// The unique module designs
    std::vector<module_digitization_config> designs;
    /// Map from detray module ID to index into designs
    std::unordered_map<uint64_t, int> id_to_design_index;

    /// Convenience lookup
    const module_digitization_config* get(uint64_t detray_id) const {
        auto it = id_to_design_index.find(detray_id);
        if (it == id_to_design_index.end()) return nullptr;
        return &designs[it->second];
    }
};

}  // namespace traccc