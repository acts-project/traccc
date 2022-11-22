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

namespace traccc::io::csv {

/// Type to read information into about the detector modules/surfaces
struct surface {

    uint64_t geometry_id = 0;
    float cx = 0., cy = 0., cz = 0.;
    float rot_xu = 0., rot_xv = 0., rot_xw = 0.;
    float rot_yu = 0., rot_yv = 0., rot_yw = 0.;
    float rot_zu = 0., rot_zv = 0., rot_zw = 0.;

    // geometry_id,hit_id,channel0,channel1,timestamp,value
    DFE_NAMEDTUPLE(surface, geometry_id, cx, cy, cz, rot_xu, rot_xv, rot_xw,
                   rot_yu, rot_yv, rot_yw, rot_zu, rot_zv, rot_zw);

};  // struct surface

}  // namespace traccc::io::csv
