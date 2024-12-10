/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/indexing.hpp"
#include "detray/io/common/detail/grid_writer.hpp"
#include "detray/io/frontend/payloads.hpp"

// System include(s)
#include <string_view>

namespace detray::io {

/// @brief Surface grid writer backend
class surface_grid_writer : public detail::grid_writer {

    using base_type = detail::grid_writer;
    using grid_writer_t = base_type;

    public:
    /// Tag the writer as "surface_grids"
    static constexpr std::string_view tag = "surface_grids";

    /// Same constructors for this class as for base_type
    using base_type::base_type;

    /// Convert the header information into its payload
    template <typename detector_t>
    static auto write_header(const detector_t& det,
                             const std::string_view det_name) {

        return grid_writer_t::write_header(tag, det.accelerator_store(),
                                           det_name);
    }

    /// Convert the grid collections of a detector @param det into their io
    /// payload
    template <typename detector_t>
    static detector_grids_payload<std::size_t, io::accel_id> convert(
        const detector_t& det, const typename detector_t::name_map&) {

        using surface_desc_t = typename detector_t::surface_type;

        detector_grids_payload<std::size_t, io::accel_id> grids_data;

        for (const auto& vol_desc : det.volumes()) {
            // Links to all acceleration data structures in the volume
            const auto& multi_link = vol_desc.accel_link();

            // How to convert the surface descriptors in the grid
            auto sf_converter = [&vol_desc = std::as_const(vol_desc)](
                                    const surface_desc_t& sf_desc) {
                return vol_desc.to_local_sf_index(sf_desc.index());
            };

            // Start a 1, because the first acceleration structure is always
            // the brute force method
            for (dindex i = 1u; i < multi_link.size(); ++i) {
                const auto& acc_link = multi_link[i];
                // Don't look at empty links
                if (acc_link.is_invalid()) {
                    continue;
                }

                // Generate the payload
                grid_writer_t::convert(det.accelerator_store(), acc_link,
                                       vol_desc.index(), vol_desc.index(),
                                       grids_data, sf_converter);
            }
        }

        return grids_data;
    }
};

}  // namespace detray::io
