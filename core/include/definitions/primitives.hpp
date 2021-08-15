/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <stdint.h>

#include <vecmem/containers/static_array.hpp>

#include "definitions/algebra.hpp"

namespace traccc {

using geometry_id = uint64_t;
using event_id = uint64_t;
using channel_id = unsigned int;

using vector2 = array::array<scalar, 2>;
using point2 = array::array<scalar, 2>;
using variance2 = array::array<scalar, 2>;
using point3 = array::array<scalar, 3>;
using vector3 = array::array<scalar, 3>;
using variance3 = array::array<scalar, 3>;
using transform3 = array::transform3;

}  // namespace traccc
