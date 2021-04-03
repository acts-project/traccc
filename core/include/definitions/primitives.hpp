/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <array>
#include <stdint.h>

namespace traccc {

    using scalar = float;
    using geometry_id = uint64_t;
    using event_id = uint64_t;

    using vector2 = std::array<scalar, 2>;
    using point2 = std::array<scalar, 2>;
    using variance2 = std::array<scalar, 2>;
    using point3 = std::array<scalar, 3>;
    using vector3 = std::array<scalar, 3>;
    using variance3 = std::array<scalar, 3>;
}
