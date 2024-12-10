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
#include "detray/io/json/json_algebra_io.hpp"
#include "detray/io/json/json_common_io.hpp"
#include "detray/io/json/json_grids_io.hpp"

// System include(s)
#include <array>
#include <optional>
#include <vector>

/// @brief  The detray JSON I/O is written in such a way that it
/// can read/write ACTS files that are written with the Detray
/// JSON I/O extension
namespace detray::io {

inline void to_json(nlohmann::ordered_json& j, const geo_header_payload& h) {
    j["common"] = h.common;

    if (h.sub_header.has_value()) {
        const auto& geo_sub_header = h.sub_header.value();
        j["volume_count"] = geo_sub_header.n_volumes;
        j["surface_count"] = geo_sub_header.n_surfaces;
    }
}

inline void from_json(const nlohmann::ordered_json& j, geo_header_payload& h) {
    h.common = j["common"];

    if (j.find("volume_count") != j.end() &&
        j.find("surface_count") != j.end()) {
        h.sub_header.emplace();
        auto& geo_sub_header = h.sub_header.value();
        geo_sub_header.n_volumes = j["volume_count"];
        geo_sub_header.n_surfaces = j["surface_count"];
    }
}

inline void to_json(nlohmann::ordered_json& j, const mask_payload& m) {
    j["shape"] = static_cast<unsigned int>(m.shape);
    j["volume_link"] = m.volume_link;
    j["boundaries"] = m.boundaries;
}

inline void from_json(const nlohmann::ordered_json& j, mask_payload& m) {
    m.shape = static_cast<mask_payload::mask_shape>(j["shape"]);
    m.volume_link = j["volume_link"];
    m.boundaries = j["boundaries"].get<std::vector<real_io>>();
}

inline void to_json(nlohmann::ordered_json& j, const surface_payload& s) {
    if (s.barcode.has_value()) {
        j["barcode"] = s.barcode.value();
    }
    j["type"] = static_cast<unsigned int>(s.type);
    j["source"] = s.source;
    j["transform"] = s.transform;
    j["mask"] = s.mask;
    if (s.material.has_value()) {
        j["material"] = s.material.value();
    }
    if (s.index_in_coll.has_value()) {
        j["index_in_coll"] = s.index_in_coll.value();
    }
}

inline void from_json(const nlohmann::ordered_json& j, surface_payload& s) {
    if (j.find("barcode") != j.end()) {
        s.barcode = j["barcode"];
    }
    s.type = static_cast<detray::surface_id>(j["type"]);
    s.source = j["source"];
    s.transform = j["transform"];
    s.mask = j["mask"];
    if (j.find("material") != j.end()) {
        s.material = j["material"];
    }
    if (j.find("index_in_coll") != j.end()) {
        s.index_in_coll = j["index_in_coll"];
    }
}

inline void to_json(nlohmann::ordered_json& j, const volume_payload& v) {
    j["name"] = v.name;
    j["index"] = v.index;
    j["type"] = v.type;
    j["transform"] = v.transform;
    nlohmann::ordered_json sjson;
    for (const auto& s : v.surfaces) {
        sjson.push_back(s);
    }
    j["surfaces"] = sjson;
    if (v.acc_links.has_value() && !v.acc_links.value().empty()) {
        nlohmann::ordered_json ljson;
        for (const auto& al : v.acc_links.value()) {
            ljson.push_back(al);
        }
        j["acc_links"] = ljson;
    }
}

inline void from_json(const nlohmann::ordered_json& j, volume_payload& v) {
    v.name = j["name"];
    v.index = j["index"];
    v.type = j["type"];
    v.transform = j["transform"];
    for (auto js : j["surfaces"]) {
        surface_payload s = js;
        v.surfaces.push_back(s);
    }
    if (j.find("acc_links") != j.end()) {
        v.acc_links.emplace();
        for (auto jl : j["acc_links"]) {
            acc_links_payload al = jl;
            v.acc_links->push_back(al);
        }
    }
}

inline void to_json(nlohmann::ordered_json& j, const detector_payload& d) {
    if (!d.volumes.empty()) {
        nlohmann::ordered_json jvolumes;
        for (const auto& v : d.volumes) {
            jvolumes.push_back(v);
        }
        j["volumes"] = jvolumes;
        if (d.volume_grid.has_value()) {
            j["volume_grid"] = d.volume_grid.value();
        }
    }
}

inline void from_json(const nlohmann::ordered_json& j, detector_payload& d) {
    if (j.find("volumes") != j.end()) {
        for (auto jvolume : j["volumes"]) {
            d.volumes.push_back(jvolume);
        }
    }
    // @TODO Put back once volume grids can be read
    /*if (j.find("volume_grid") != j.end()) {
        d.volume_grid = j["volume_grid"];
    }*/
}

}  // namespace detray::io
