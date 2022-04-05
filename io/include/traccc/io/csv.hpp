/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/geometry/geometry.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// DFE include(s).
#include <dfe/dfe_io_dsv.hpp>
#include <dfe/dfe_namedtuple.hpp>

// System include(s).
#include <cassert>
#include <climits>
#include <fstream>
#include <iostream>
#include <map>

/// reader
namespace traccc {

/// reader

struct csv_cell {

    uint64_t geometry_id = 0;
    uint64_t hit_id = 0;
    channel_id channel0 = 0;
    channel_id channel1 = 0;
    scalar timestamp = 0.;
    scalar value = 0.;

    // geometry_id,hit_id,channel0,channel1,timestamp,value
    DFE_NAMEDTUPLE(csv_cell, geometry_id, hit_id, channel0, channel1, timestamp,
                   value);
};

using cell_reader = dfe::NamedTupleCsvReader<csv_cell>;

struct csv_fatras_hit {
    uint64_t particle_id = 0;
    uint64_t geometry_id = 0;
    scalar tx = 0;
    scalar ty = 0;
    scalar tz = 0;
    scalar tt = 0;
    scalar tpx = 0;
    scalar tpy = 0;
    scalar tpz = 0;
    scalar te = 0;
    scalar deltapx = 0;
    scalar deltapy = 0;
    scalar deltapz = 0;
    scalar deltae = 0;
    uint64_t index = 0;

    DFE_NAMEDTUPLE(csv_fatras_hit, particle_id, geometry_id, tx, ty, tz, tt,
                   tpx, tpy, tpz, te, deltapx, deltapy, deltapz, deltae, index);
};

using fatras_hit_reader = dfe::NamedTupleCsvReader<csv_fatras_hit>;

struct csv_particle {
    uint64_t particle_id = 0;
    int particle_type = 0;
    int process = 0;
    scalar vx;
    scalar vy;
    scalar vz;
    scalar vt;
    scalar px;
    scalar py;
    scalar pz;
    scalar m;
    scalar q;

    DFE_NAMEDTUPLE(csv_particle, particle_id, particle_type, process, vx, vy,
                   vz, vt, px, py, pz, m, q);
};

using fatras_particle_reader = dfe::NamedTupleCsvReader<csv_particle>;

/// writer

struct csv_measurement {

    uint64_t geometry_id = 0;
    std::string local_key = "";
    scalar local0 = 0.;
    scalar local1 = 0.;
    scalar phi = 0.;
    scalar theta = 0.;
    scalar time = 0.;
    scalar var_local0 = 0.;
    scalar var_local1 = 0.;
    scalar var_phi = 0.;
    scalar var_theta = 0.;
    scalar var_time = 0.;

    DFE_NAMEDTUPLE(csv_measurement, geometry_id, local0, local1, phi, theta,
                   time, var_local0, var_local1, var_phi, var_theta, var_time);
};

using measurement_reader = dfe::NamedTupleCsvReader<csv_measurement>;
using measurement_writer = dfe::NamedTupleCsvWriter<csv_measurement>;

struct csv_spacepoint {

    uint64_t geometry_id = 0;
    scalar x, y, z;

    DFE_NAMEDTUPLE(csv_spacepoint, geometry_id, x, y, z);
};

using spacepoint_writer = dfe::NamedTupleCsvWriter<csv_spacepoint>;

struct csv_seed {
    scalar weight;
    scalar z_vertex;
    scalar x_b, y_b, z_b;
    scalar varR_b, varZ_b;
    scalar x_m, y_m, z_m;
    scalar varR_m, varZ_m;
    scalar x_t, y_t, z_t;
    scalar varR_t, varZ_t;

    DFE_NAMEDTUPLE(csv_seed, weight, z_vertex, x_b, y_b, z_b, varR_b, varZ_b,
                   x_m, y_m, z_m, varR_m, varZ_m, x_t, y_t, z_t, varR_t,
                   varZ_t);
};

using seed_writer = dfe::NamedTupleCsvWriter<csv_seed>;

struct csv_bound_track_parameters {
    scalar loc0;
    scalar loc1;
    scalar theta;
    scalar phi;
    scalar qoverp;
    scalar time;
    scalar cov00;
    scalar cov01, cov11;
    scalar cov02, cov12, cov22;
    scalar cov03, cov13, cov23, cov33;
    scalar cov04, cov14, cov24, cov34, cov44;
    scalar cov05, cov15, cov25, cov35, cov45, cov55;
    unsigned int surface_id;

    DFE_NAMEDTUPLE(csv_bound_track_parameters, loc0, loc1, theta, phi, qoverp,
                   time, cov00, cov01, cov11, cov02, cov12, cov22, cov03, cov13,
                   cov23, cov33, cov04, cov14, cov24, cov34, cov44, cov05,
                   cov15, cov25, cov35, cov45, cov55, surface_id);
};

using bound_track_parameters_writer =
    dfe::NamedTupleCsvWriter<csv_bound_track_parameters>;

struct csv_surface {

    uint64_t geometry_id = 0;
    scalar cx, cy, cz;
    scalar rot_xu, rot_xv, rot_xw;
    scalar rot_yu, rot_yv, rot_yw;
    scalar rot_zu, rot_zv, rot_zw;

    // geometry_id,hit_id,channel0,channel1,timestamp,value
    DFE_NAMEDTUPLE(csv_surface, geometry_id, cx, cy, cz, rot_xu, rot_xv, rot_xw,
                   rot_yu, rot_yv, rot_yw, rot_zu, rot_zv, rot_zw);
};

using surface_reader = dfe::NamedTupleCsvReader<csv_surface>;

/// Read the geometry information per module and fill into a map
///
/// @param sreader The surface reader type
inline std::map<geometry_id, transform3> read_surfaces(
    surface_reader& sreader) {

    std::map<geometry_id, transform3> transform_map;
    csv_surface iosurface;
    while (sreader.read(iosurface)) {

        geometry_id module = iosurface.geometry_id;

        vector3 t{iosurface.cx, iosurface.cy, iosurface.cz};
        vector3 x{iosurface.rot_xu, iosurface.rot_yu, iosurface.rot_zu};
        vector3 z{iosurface.rot_xw, iosurface.rot_yw, iosurface.rot_zw};

        transform_map.insert({module, transform3{t, z, x}});
    }
    return transform_map;
}

/// Read the collection of cells per module and fill into a collection
///
/// @param creader The cellreader type
/// @param resource The memory resource to use for the return value
/// @param tfmap the (optional) transform map
/// @param max_cells the (optional) maximum number of cells to be read in
inline host_cell_container read_cells(
    cell_reader& creader, vecmem::memory_resource& resource,
    const traccc::geometry* tfmap = nullptr,
    unsigned int max_cells = std::numeric_limits<unsigned int>::max()) {

    uint64_t reference_id = 0;
    host_cell_container result(&resource);

    bool first_line_read = false;
    unsigned int read_cells = 0;
    csv_cell iocell;
    host_cell_collection cells(&resource);
    cell_module module;
    while (creader.read(iocell)) {

        if (first_line_read and iocell.geometry_id != reference_id) {
            // Complete the information
            if (tfmap != nullptr) {
                if (tfmap->contains(iocell.geometry_id)) {
                    module.placement = (*tfmap)[iocell.geometry_id];
                }
            }
            // Sort in column major order
            std::sort(cells.begin(), cells.end(),
                      [](const auto& a, const auto& b) {
                          return a.channel1 < b.channel1;
                      });
            result.push_back(std::move(module), std::move(cells));
            // Clear for next round
            cells = host_cell_collection(&resource);
            module = cell_module();
        }
        first_line_read = true;
        reference_id = static_cast<uint64_t>(iocell.geometry_id);

        module.module = reference_id;
        module.range0[0] = std::min(module.range0[0], iocell.channel0);
        module.range0[1] = std::max(module.range0[1], iocell.channel0);
        module.range1[0] = std::min(module.range1[0], iocell.channel1);
        module.range1[1] = std::max(module.range1[1], iocell.channel1);

        cells.push_back(cell{iocell.channel0, iocell.channel1, iocell.value,
                             iocell.timestamp});
        if (++read_cells >= max_cells) {
            break;
        }
    }

    // Clean up after loop
    // Sort in column major order
    std::sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) {
        return a.channel1 < b.channel1;
    });

    result.push_back(std::move(module), std::move(cells));

    return result;
}

/// Read the collection of cells per module and fill into a collection
/// of truth clusters.
///
/// @param creader The cellreader type
/// @param resource The memory resource to use for the return value
/// @param tfmap the (optional) transform map
/// @param max_clusters the (optional) maximum number of cells to be read in
inline std::vector<host_cluster_container> read_truth_clusters(
    cell_reader& creader, vecmem::memory_resource& resource,
    const std::map<geometry_id, transform3>& tfmap = {},
    unsigned int max_cells = std::numeric_limits<unsigned int>::max()) {

    // Reference for switching the container
    uint64_t reference_id = 0;
    std::vector<host_cluster_container> clusters;
    // Reference for switching the cluster
    uint64_t truth_id = std::numeric_limits<uint64_t>::max();

    bool first_line_read = false;
    unsigned int read_cells = 0;
    csv_cell iocell;
    host_cluster_container truth_clusters(&resource);
    vecmem::vector<cell> truth_cells;

    while (creader.read(iocell)) {

        if (first_line_read and iocell.geometry_id != reference_id) {
            // Complete the information
            if (not tfmap.empty()) {
                auto tfentry = tfmap.find(iocell.geometry_id);
                if (tfentry != tfmap.end()) {
                    for (auto& truth_cl_id : truth_clusters.get_headers())
                        truth_cl_id.placement = tfentry->second;
                }
            }

            // Sort in column major order
            clusters.push_back(truth_clusters);
            // Clear for next round
            truth_clusters = host_cluster_container(&resource);
        }

        if (first_line_read and truth_id != iocell.hit_id) {
            truth_clusters.get_items().push_back({truth_cells});
            truth_cells.clear();
        }
        truth_cells.push_back(cell{iocell.channel0, iocell.channel1,
                                   iocell.value, iocell.timestamp});

        first_line_read = true;
        truth_id = iocell.hit_id;
        reference_id = static_cast<uint64_t>(iocell.geometry_id);

        if (++read_cells >= max_cells) {
            break;
        }
    }

    return clusters;
}
/// Read the collection of measurements per module and fill into a collection
///
/// @param hreader The measurement reader type
/// @param resource The memory resource to use for the return value
inline host_measurement_container read_measurements(
    measurement_reader& mreader, vecmem::memory_resource& resource,
    const std::map<geometry_id, transform3>& tfmap = {},
    unsigned int max_measurements = std::numeric_limits<unsigned int>::max()) {

    uint64_t reference_id = 0;
    host_measurement_container result = {
        host_measurement_container::header_vector(&resource),
        host_measurement_container::item_vector(&resource)};

    bool first_line_read = false;
    unsigned int read_measurements = 0;
    csv_measurement iomeasurement;
    host_measurement_collection measurements(&resource);
    cell_module module;
    while (mreader.read(iomeasurement)) {
        if (first_line_read and iomeasurement.geometry_id != reference_id) {
            // Complete the information
            if (not tfmap.empty()) {
                auto tfentry = tfmap.find(iomeasurement.geometry_id);
                if (tfentry != tfmap.end()) {
                    module.placement = tfentry->second;
                }
            }

            result.push_back(std::move(module), std::move(measurements));

            // Clear for next round
            measurements = host_measurement_collection(&resource);
            module = cell_module();
        }
        first_line_read = true;
        reference_id = static_cast<uint64_t>(iomeasurement.geometry_id);

        module.module = reference_id;

        measurements.push_back(
            measurement{{iomeasurement.local0, iomeasurement.local1},
                        {iomeasurement.var_local0, iomeasurement.var_local1}});
        if (++read_measurements >= max_measurements) {
            break;
        }
    }

    result.push_back(std::move(module), std::move(measurements));

    return result;
}

/// Read the collection of hits per module and fill into a collection
///
/// @param hreader The hit reader type
/// @param resource The memory resource to use for the return value
inline host_spacepoint_container read_hits(
    fatras_hit_reader& hreader, vecmem::memory_resource& resource,
    const traccc::geometry* tfmap = nullptr,
    unsigned int max_hits = std::numeric_limits<unsigned int>::max()) {
    host_spacepoint_container result(&resource);

    unsigned int read_hits = 0;
    csv_fatras_hit iohit;

    host_spacepoint_collection spacepoints(&resource);

    while (hreader.read(iohit)) {
        geometry_id geom_id = iohit.geometry_id;
        auto placement = (*tfmap)[geom_id];

        point3 position({iohit.tx, iohit.ty, iohit.tz});
        auto local = placement.point_to_local(position);
        measurement m({point2({local[0], local[1]}), variance2({0., 0.})});
        spacepoint sp({position, m});

        const host_spacepoint_container::header_vector& headers =
            result.get_headers();

        auto it = std::find(headers.begin(), headers.end(), geom_id);

        if (it == headers.end()) {
            result.push_back(geom_id, vecmem::vector<spacepoint>({sp}));
        } else {
            auto idx = it - headers.begin();
            result.at(idx).items.push_back(sp);
        }

        if (++read_hits >= max_hits) {
            break;
        }
    }

    return result;
}

}  // namespace traccc
