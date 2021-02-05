/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <array>

namespace traccc {

    using scalar = float;

    using geometry_id = unsigned int;

    using vector2 = std::array<scalar, 2>;
    using point2 = std::array<scalar, 2>;
    using covariance2 = std::array<scalar, 2>;
    using point3 = std::array<scalar, 3>;
    using vector3 = std::array<scalar, 3>;
    using covariance3 = std::array<scalar, 3>;

}
