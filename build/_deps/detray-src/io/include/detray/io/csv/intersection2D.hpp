/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/io/utils/create_path.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/utils/ranges.hpp"

// DFE include(s).
#include <dfe/dfe_io_dsv.hpp>
#include <dfe/dfe_namedtuple.hpp>

// System include(s).
#include <cstdint>
#include <filesystem>

namespace detray::io::csv {

/// Type to read the data of a line-surface intersection
struct intersection2D {

    unsigned int track_id = 0;
    std::uint64_t identifier = 0ul;
    unsigned int type = 0u;
    unsigned int transform_index = 0u;
    unsigned int mask_id = 0u;
    unsigned int mask_index = 0u;
    unsigned int material_id = 0u;
    unsigned int material_index = 0u;
    double l0 = 0.;
    double l1 = 0.;
    double path = 0.;
    unsigned int volume_link = 0u;
    int direction = 0;
    int status = 0;

    DFE_NAMEDTUPLE(intersection2D, track_id, identifier, type, transform_index,
                   mask_id, mask_index, material_id, material_index, l0, l1,
                   path, volume_link, direction, status);
};

/// Read intersections from csv file
/// @returns vector of intersections
template <typename detector_t>
inline auto read_intersection2D(const std::string &file_name) {

    using algebra_t = typename detector_t::algebra_type;
    using scalar_t = typename detector_t::scalar_type;
    using surface_t = typename detector_t::surface_type;
    using nav_link_t = typename surface_t::navigation_link;
    using mask_link_t = typename surface_t::mask_link;
    using material_link_t = typename surface_t::material_link;
    using mask_id_t = typename detector_t::masks::id;
    using material_id_t = typename detector_t::materials::id;

    using intersection_t = detray::intersection2D<surface_t, algebra_t, true>;

    dfe::NamedTupleCsvReader<io::csv::intersection2D> inters_reader(file_name);

    io::csv::intersection2D inters_data{};
    std::vector<std::vector<intersection_t>> intersections_per_track;

    // Read the data
    while (inters_reader.read(inters_data)) {

        // Add new intersection to correct track
        auto trk_index{static_cast<dindex>(inters_data.track_id)};
        while (intersections_per_track.size() <= trk_index) {
            intersections_per_track.push_back({});
        }

        // Read the intersection
        intersection_t inters{};

        mask_link_t mask_link{static_cast<mask_id_t>(inters_data.mask_id),
                              inters_data.mask_index};
        material_link_t material_link{
            static_cast<material_id_t>(inters_data.material_id),
            inters_data.material_index};
        inters.sf_desc = {inters_data.transform_index, mask_link, material_link,
                          dindex_invalid, surface_id::e_unknown};
        inters.sf_desc.set_barcode(geometry::barcode{inters_data.identifier});
        inters.local = {static_cast<scalar_t>(inters_data.l0),
                        static_cast<scalar_t>(inters_data.l1), 0.f};
        inters.path = static_cast<scalar_t>(inters_data.path);
        inters.volume_link = static_cast<nav_link_t>(inters_data.volume_link);
        inters.direction = static_cast<bool>(inters_data.direction);
        inters.status = static_cast<bool>(inters_data.status);

        // Add to collection
        intersections_per_track[trk_index].push_back(inters);
    }

    // Check the result
    if (intersections_per_track.empty()) {
        throw std::invalid_argument(
            "ERROR: csv reader: Failed to read intersection data");
    }

    return intersections_per_track;
}

/// Write intersections to csv file
template <typename intersection_t>
inline void write_intersection2D(
    const std::string &file_name,
    const std::vector<std::vector<intersection_t>> &intersections_per_track,
    const bool replace = true) {

    // Don't write over existing data
    std::string inters_file_name{file_name};
    if (!replace && io::file_exists(file_name)) {
        inters_file_name = io::alt_file_name(file_name);
    } else {
        // Make sure the output directories exit
        io::create_path(std::filesystem::path{inters_file_name}.parent_path());
    }

    dfe::NamedTupleCsvWriter<io::csv::intersection2D> inters_writer(
        inters_file_name);

    for (const auto &[track_idx, intersections] :
         detray::views::enumerate(intersections_per_track)) {

        // Skip empty traces
        if (intersections.empty()) {
            continue;
        }

        for (const auto &inters : intersections) {
            io::csv::intersection2D inters_data{};

            inters_data.track_id = track_idx;
            inters_data.identifier = inters.sf_desc.barcode().value();
            inters_data.type =
                static_cast<unsigned int>(inters.sf_desc.barcode().id());
            inters_data.transform_index = inters.sf_desc.transform();
            inters_data.mask_id =
                static_cast<unsigned int>(inters.sf_desc.mask().id());
            inters_data.mask_index = inters.sf_desc.mask().index();
            inters_data.material_id =
                static_cast<unsigned int>(inters.sf_desc.material().id());
            inters_data.material_index = inters.sf_desc.material().index();
            inters_data.l0 = inters.local[0];
            inters_data.l1 = inters.local[1];
            inters_data.path = inters.path;
            inters_data.volume_link = inters.volume_link;
            inters_data.direction = static_cast<int>(inters.direction);
            inters_data.status = static_cast<int>(inters.status);

            inters_writer.append(inters_data);
        }
    }
}

}  // namespace detray::io::csv
