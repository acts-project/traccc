/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

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
    uint8_t local_key = 0;
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

    DFE_NAMEDTUPLE(measurement, measurement_id, geometry_id, local_key, local0,
                   local1, phi, theta, time, var_local0, var_local1, var_phi,
                   var_theta, var_time);
};

inline traccc::measurement make_measurement_edm(
    const traccc::io::csv::measurement& csv_meas,
    std::map<geometry_id, geometry_id>* acts_to_detray_id) {

    // Construct the measurement object.
    traccc::measurement meas;
    std::array<typename transform3::size_type, 2u> indices{0u, 0u};
    meas.meas_dim = 0u;

    // Local key is a 8 bit char and first and last bit are dummy value. 2 -
    // 7th bits are for 6 bound track parameters.
    // Ex1) 0000010 or 2 -> meas dim = 1 and [loc0] active -> strip or wire
    // Ex2) 0000110 or 6 -> meas dim = 2 and [loc0, loc1] active -> pixel
    // Ex3) 0000100 or 4 -> meas dim = 1 and [loc1] active -> annulus
    for (unsigned int ipar = 0; ipar < 2u; ++ipar) {
        if (((csv_meas.local_key) & (1 << (ipar + 1))) != 0) {

            switch (ipar) {
                case e_bound_loc0: {
                    meas.local[0] = csv_meas.local0;
                    meas.variance[0] = csv_meas.var_local0;
                    indices[meas.meas_dim++] = ipar;
                }; break;
                case e_bound_loc1: {
                    meas.local[1] = csv_meas.local1;
                    meas.variance[1] = csv_meas.var_local1;
                    indices[meas.meas_dim++] = ipar;
                }; break;
            }
        }
    }

    meas.subs.set_indices(indices);
    if (acts_to_detray_id) {
        meas.surface_link = detray::geometry::barcode{
            acts_to_detray_id->at(csv_meas.geometry_id)};
    } else {
        meas.surface_link = detray::geometry::barcode{csv_meas.geometry_id};
    }

    return meas;
}

}  // namespace traccc::io::csv
