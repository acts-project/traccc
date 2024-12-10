/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/builders/detector_builder.hpp"
#include "detray/builders/grid_builder.hpp"
#include "detray/io/common/detail/grid_reader.hpp"
#include "detray/io/frontend/payloads.hpp"

// System include(s)
#include <string_view>

namespace detray::io {

/// @brief Surface grid reader backend
template <class surface_descriptor_t,
          typename CAP = std::integral_constant<std::size_t, 9>,
          typename DIM = std::integral_constant<std::size_t, 2>>
class surface_grid_reader
    : public detail::grid_reader<surface_descriptor_t, grid_builder, CAP, DIM> {

    using grid_reader_t =
        detail::grid_reader<surface_descriptor_t, grid_builder, CAP, DIM>;
    using base_type = grid_reader_t;

    public:
    /// Tag the reader as "surface_grids"
    static constexpr std::string_view tag = "surface_grids";

    /// Same constructors for this class as for base_type
    using base_type::base_type;

    /// Convert the detector grids @param grids_data from their IO
    /// payload
    template <typename detector_t>
    static void convert(
        detector_builder<typename detector_t::metadata, volume_builder>
            &det_builder,
        typename detector_t::name_map &,
        const detector_grids_payload<std::size_t, io::accel_id> &grids_data) {

        grid_reader_t::template convert<detector_t>(det_builder, grids_data);
    }
};

}  // namespace detray::io
