/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/bfield/magnetic_field_types.hpp"

namespace traccc::alpaka {

/// @brief the standard list of Alpaka bfield types to support
template <typename scalar_t>
using bfield_type_list = std::tuple<const_bfield_backend_t<scalar_t>,
                                    host::inhom_bfield_backend_t<scalar_t>>;

}  // namespace traccc::alpaka
