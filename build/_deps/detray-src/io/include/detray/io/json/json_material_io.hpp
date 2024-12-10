/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/io/frontend/payloads.hpp"
#include "detray/io/json/json.hpp"
#include "detray/io/json/json_common_io.hpp"

// System include(s)
#include <array>

/// @brief  The detray JSON I/O is written in such a way that it
/// can read/write ACTS files that are written with the Detray
/// JSON I/O extension
namespace detray::io {

inline void to_json(nlohmann::ordered_json& j,
                    const homogeneous_material_header_payload& h) {
    j["common"] = h.common;

    if (h.sub_header.has_value()) {
        const auto& mat_sub_header = h.sub_header.value();
        j["slab_count"] = mat_sub_header.n_slabs;
        j["rod_count"] = mat_sub_header.n_rods;
    }
}

inline void from_json(const nlohmann::ordered_json& j,
                      homogeneous_material_header_payload& h) {
    h.common = j["common"];

    if (j.find("slab_count") != j.end() && j.find("rod_count") != j.end()) {
        h.sub_header.emplace();
        auto& mat_sub_header = h.sub_header.value();
        mat_sub_header.n_slabs = j["slab_count"];
        mat_sub_header.n_rods = j["rod_count"];
    }
}

inline void to_json(nlohmann::ordered_json& j, const material_payload& m) {
    j["params"] = m.params;
}

inline void from_json(const nlohmann::ordered_json& j, material_payload& m) {
    m.params = j["params"].get<std::array<real_io, 7>>();
}

inline void to_json(nlohmann::ordered_json& j, const material_slab_payload& m) {
    j["type"] = m.type;
    j["surface_idx"] = m.surface;
    j["thickness"] = m.thickness;
    j["material"] = m.mat;
    if (m.index_in_coll.has_value()) {
        j["index_in_coll"] = m.index_in_coll.value();
    }
}

inline void from_json(const nlohmann::ordered_json& j,
                      material_slab_payload& m) {
    m.type = j["type"];
    m.surface = j["surface_idx"];
    m.thickness = j["thickness"];
    m.mat = j["material"];
    if (j.find("index_in_coll") != j.end()) {
        m.index_in_coll = j["index_in_coll"];
    }
}

inline void to_json(nlohmann::ordered_json& j,
                    const material_volume_payload& mv) {
    j["volume_link"] = mv.volume_link;

    if (!mv.mat_slabs.empty()) {
        nlohmann::ordered_json jmats;
        for (const auto& m : mv.mat_slabs) {
            jmats.push_back(m);
        }
        j["material_slabs"] = jmats;
    }
    if (mv.mat_rods.has_value() && !mv.mat_rods->empty()) {
        nlohmann::ordered_json jmats;
        for (const auto& m : mv.mat_rods.value()) {
            jmats.push_back(m);
        }
        j["material_rods"] = jmats;
    }
}

inline void from_json(const nlohmann::ordered_json& j,
                      material_volume_payload& mv) {
    mv.volume_link = j["volume_link"];

    if (j.find("material_slabs") != j.end()) {
        for (auto jmats : j["material_slabs"]) {
            material_slab_payload mslp = jmats;
            mv.mat_slabs.push_back(mslp);
        }
    }
    if (j.find("material_rods") != j.end()) {
        mv.mat_rods.emplace();
        for (auto jmats : j["material_rods"]) {
            material_slab_payload mslp = jmats;
            mv.mat_rods->push_back(mslp);
        }
    }
}

inline void to_json(nlohmann::ordered_json& j,
                    const detector_homogeneous_material_payload& d) {
    if (!d.volumes.empty()) {
        nlohmann::ordered_json jmats;
        for (const auto& m : d.volumes) {
            jmats.push_back(m);
        }
        j["volumes"] = jmats;
    }
}

inline void from_json(const nlohmann::ordered_json& j,
                      detector_homogeneous_material_payload& d) {
    if (j.find("volumes") != j.end()) {
        for (auto jvolume : j["volumes"]) {
            d.volumes.push_back(jvolume);
        }
    }
}

}  // namespace detray::io
