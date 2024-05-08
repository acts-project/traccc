/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

namespace traccc {

/// A very basic pixel segmentation with
/// a minimum corner and ptich x/y
///
/// No checking on out of bounds done
struct pixel_data {

    scalar min_center_x = 0.;
    scalar min_center_y = 0.;
    scalar pitch_x = 1.;
    scalar pitch_y = 1.;
    char dimension = 2;

    TRACCC_HOST_DEVICE
    vector2 get_pitch() const { return {pitch_x, pitch_y}; };
};

}  // namespace traccc
