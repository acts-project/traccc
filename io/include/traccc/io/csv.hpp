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
#include "traccc/edm/container.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/geometry/digitization_config.hpp"
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
#include <stdexcept>

/// reader
namespace traccc {

/// reader

struct csv_meas_hit_id {

    uint64_t measurement_id = 0;
    uint64_t hit_id = 0;

    // geometry_id,hit_id,channel0,channel1,timestamp,value
    DFE_NAMEDTUPLE(csv_meas_hit_id, measurement_id, hit_id);
};

using meas_hit_id_reader = dfe::NamedTupleCsvReader<csv_meas_hit_id>;

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

}  // namespace traccc
