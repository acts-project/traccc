/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/builders/detector_builder.hpp"
#include "detray/builders/material_map_builder.hpp"
#include "detray/io/common/detail/basic_converter.hpp"
#include "detray/io/common/detail/type_info.hpp"
#include "detray/io/common/homogeneous_material_reader.hpp"
#include "detray/io/frontend/payloads.hpp"
#include "detray/materials/material_slab.hpp"
#include "detray/utils/type_list.hpp"

// System include(s)
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace detray::io {

/// @brief Material map reader backend
template <typename DIM = std::integral_constant<std::size_t, 2u>>
class material_map_reader {

    using material_reader_t = homogeneous_material_reader;

    public:
    static constexpr std::size_t dim{DIM()};

    using bin_index_type = axis::multi_bin<dim>;

    /// Tag the reader as "material_maps"
    static constexpr std::string_view tag = "material_maps";

    /// Convert the material grids @param grids_data from their IO
    /// payload
    template <typename detector_t>
    static void convert(detector_builder<typename detector_t::metadata,
                                         volume_builder> &det_builder,
                        typename detector_t::name_map &,
                        detector_grids_payload<material_slab_payload,
                                               io::material_id> &&grids_data) {

        using scalar_t = typename detector_t::scalar_type;
        using mat_factory_t = material_map_factory<detector_t, bin_index_type>;
        using mat_data_t = typename mat_factory_t::data_type;
        using mat_id = typename detector_t::materials::id;

        // Convert the material volume by volume
        for (const auto &[vol_idx, mat_grids] : grids_data.grids) {

            if (!det_builder.has_volume(vol_idx)) {
                std::stringstream err_stream;
                err_stream << "Volume " << vol_idx << ": "
                           << "Cannot build material map for volume "
                           << "(volume not registered in detector builder)";
                throw std::invalid_argument(err_stream.str());
            }

            // Decorate the current volume builder with material maps
            auto vm_builder =
                det_builder
                    .template decorate<material_map_builder<detector_t, dim>>(
                        static_cast<dindex>(vol_idx));

            // Add the material data to the factory
            auto mat_factory = std::make_shared<
                material_map_factory<detector_t, bin_index_type>>();

            // Convert the material grid of each surface
            for (const auto &grid_data : mat_grids) {

                mat_id map_id = convert<io::material_id::n_mats, detector_t>(
                    grid_data.grid_link.type);

                // Get the number of bins per axis
                std::vector<std::size_t> n_bins{};
                for (const auto &axis_data : grid_data.axes) {
                    n_bins.push_back(axis_data.bins);
                }

                // Get the axis spans
                std::vector<std::vector<scalar_t>> axis_spans = {};
                for (const auto &axis_data : grid_data.axes) {
                    axis_spans.push_back(
                        {static_cast<scalar_t>(axis_data.edges.front()),
                         static_cast<scalar_t>(axis_data.edges.back())});
                }

                // Get the local bin indices and the material parametrization
                std::vector<bin_index_type> loc_bins{};
                mat_data_t mat_data{
                    detail::basic_converter::convert(grid_data.owner_link)};
                for (const auto &bin_data : grid_data.bins) {

                    assert(dim == bin_data.loc_index.size() &&
                           "Dimension of local bin indices in input file does "
                           "not match material grid dimension");

                    // The local bin indices for the bin to be filled
                    bin_index_type mbin;
                    for (const auto &[i, bin_idx] :
                         detray::views::enumerate(bin_data.loc_index)) {
                        mbin[i] = bin_idx;
                    }
                    loc_bins.push_back(std::move(mbin));

                    // For now assume surfaces ids as the only grid input
                    for (const auto &slab_data : bin_data.content) {
                        mat_data.append(
                            material_reader_t::template convert<scalar_t>(
                                slab_data));
                    }
                }

                mat_factory->add_material(
                    map_id, std::move(mat_data), std::move(n_bins),
                    std::move(axis_spans), std::move(loc_bins));
            }

            // Add the material maps to the volume
            vm_builder->add_surfaces(mat_factory);
        }
    }

    private:
    /// Get the detector material id from the payload material type id
    template <io::material_id I, typename detector_t>
    static typename detector_t::materials::id convert(io::material_id type_id) {

        /// Gets compile-time mask information
        using map_info = detail::mat_map_info<I, detector_t>;

        // Material id of map data found
        if (type_id == I) {
            // Get the corresponding material id for this detector
            return map_info::value;
        }
        // Test next material type id
        constexpr int current_id{static_cast<int>(I)};
        if constexpr (current_id > 0) {
            return convert<static_cast<io::material_id>(current_id - 1),
                           detector_t>(type_id);
        } else {
            return detector_t::materials::id::e_none;
        }
    }
};

}  // namespace detray::io
