/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// DFE include(s).
#include <dfe/dfe_namedtuple.hpp>

// System include(s).
#include <cstdint>
#include <string>

namespace traccc::io::csv {

/// Type used in reading CSV measurement data into memory
struct measurement {

    uint64_t measurement_id = 0;
    uint64_t geometry_id = 0;
    std::string local_key = "";
    float local0 = 0.;
    float local1 = 0.;
    float phi = 0.;
    float theta = 0.;
    float time = 0.;
    float var_local0 = 0.;
    float var_local1 = 0.;
    float var_phi = 0.;
    float var_theta = 0.;
    float var_time = 0.;

    DFE_NAMEDTUPLE(measurement, measurement_id, geometry_id, local0, local1,
                   phi, theta, time, var_local0, var_local1, var_phi, var_theta,
                   var_time);
};

}  // namespace traccc::io::csv
