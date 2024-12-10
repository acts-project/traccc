/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/builders/bin_fillers.hpp"
#include "detray/builders/detector_builder.hpp"
#include "detray/builders/grid_factory.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/io/common/detail/basic_converter.hpp"
#include "detray/io/common/detail/type_info.hpp"
#include "detray/io/frontend/payloads.hpp"
#include "detray/utils/ranges.hpp"
#include "detray/utils/type_list.hpp"

// System include(s)
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace detray::io::detail {

/// @brief Grid reader backend
///
/// @tparam value_t bin entry type
/// @tparam grid_builder_t the grid builder to be used
/// @tparam CAP the storage capacity of a single bin
/// @tparam DIM the dimension of the grid
/// @tparam bin_filler_t helper to fill all bins of a grid
/// @tparam serializer_t memory layout of the grid
template <typename value_t,
          template <typename, typename, typename, typename>
          class grid_builder_t,
          typename CAP = std::integral_constant<std::size_t, 0>,
          typename DIM = std::integral_constant<std::size_t, 2>,
          typename bin_filler_t = fill_by_pos,
          template <std::size_t> class serializer_t = simple_serializer>
class grid_reader {

    /// IO accelerator ids do not need to coincide with the detector ids,
    /// because they are shared with ACTS
    using acc_type = io::accel_id;

    static constexpr std::size_t dim{DIM()};
    static constexpr std::size_t bin_capacity{CAP()};

    public:
    /// Convert the detector grids @param grids_data from their IO payload
    template <typename detector_t, typename content_t, typename grid_id_t>
    static void convert(
        detector_builder<typename detector_t::metadata, volume_builder>
            &det_builder,
        const detector_grids_payload<content_t, grid_id_t> &grids_data) {

        // Convert the grids volume by volume
        for (const auto &[_, grid_data_coll] : grids_data.grids) {
            for (const auto &[i, grid_data] :
                 detray::views::enumerate(grid_data_coll)) {

                std::queue<axis::bounds> bounds;
                std::queue<axis::binning> binnings;

                for (const auto &axis_data : grid_data.axes) {
                    bounds.push(axis_data.bounds);
                    binnings.push(axis_data.binning);
                }

                // Don't start at zero, since that is the brute force method
                convert<detector_t>(bounds, binnings,
                                    std::make_pair(i + 1, grid_data),
                                    det_builder);
            }
        }
    }

    /// @brief recursively build the grid: axis bounds (open, closed, circular)
    ///
    /// @tparam bounds_ts type list that contains the bounds types that were
    ///         identified from the IO ids so far (start with empty list)
    /// @tparam binning_ts type list that contains the binning types that were
    ///         identified from the IO ids so far (start with empty list)
    ///
    /// @param bound_ids runtime queue of bounds type ids (read from file)
    /// @param binning_ids runtime queue of binning type ids (read from file)
    template <typename detector_t, typename bounds_ts = types::list<>,
              typename binning_ts = types::list<>, typename... Ts>
    static void convert(std::queue<axis::bounds> &bound_ids,
                        std::queue<axis::binning> &binning_ids, Ts &&... data) {
        using namespace axis;

        constexpr std::size_t n_bounds_types{types::size<bounds_ts>};

        // Base case: If the bounds types are filled, continue with the binnings
        if constexpr (n_bounds_types == dim) {
            return convert<detector_t, bounds_ts, binning_ts>(
                binning_ids, std::forward<Ts>(data)...);
        } else if (!bound_ids.empty()) {
            // The axis label, e.g. x, y or z by number
            constexpr auto lb{static_cast<label>(n_bounds_types)};

            const auto first_id{bound_ids.front()};
            bound_ids.pop();

            // Based on the type id, add the next bounds type to the type list
            // and continue
            switch (first_id) {
                case bounds::e_closed: {
                    using new_bounds_ts =
                        types::push_back<bounds_ts, closed<lb>>;
                    return convert<detector_t, new_bounds_ts, binning_ts>(
                        bound_ids, binning_ids, std::forward<Ts>(data)...);
                }
                case bounds::e_open: {
                    using new_bounds_ts = types::push_back<bounds_ts, open<lb>>;
                    return convert<detector_t, new_bounds_ts, binning_ts>(
                        bound_ids, binning_ids, std::forward<Ts>(data)...);
                }
                case bounds::e_circular: {
                    using new_bounds_ts =
                        types::push_back<bounds_ts, circular<lb>>;
                    return convert<detector_t, new_bounds_ts, binning_ts>(
                        bound_ids, binning_ids, std::forward<Ts>(data)...);
                }
                // Test some edge cases
                default: {
                    throw std::invalid_argument(
                        "Given type id could not be matched to a grid boundary "
                        "type: " +
                        std::to_string(static_cast<std::int64_t>(first_id)));
                    break;
                }
            }
        }
    }

    /// @brief recursively build the grid: axis binning (regular, irregular)
    ///
    /// @tparam bounds_ts type list that contains the bounds types
    /// @tparam binning_ts type list that contains the binning types that were
    ///         identified from the IO ids so far (start with empty list)
    ///
    /// @param binning_ids runtime queue of binning type ids (read from file)
    template <typename detector_t, typename bounds_ts, typename binning_ts,
              typename... Ts>
    requires(types::size<bounds_ts> ==
             dim) static void convert(std::queue<axis::binning> &binning_ids,
                                      Ts &&... data) {

        using namespace axis;

        using scalar_t = typename detector_t::scalar_type;
        using regular_binning_t = regular<host_container_types, scalar_t>;
        using irregular_binning_t = irregular<host_container_types, scalar_t>;

        // Base case: If the binning types are filled, continue with the frame
        if constexpr (types::size<binning_ts> == dim) {
            return convert<detector_t, bounds_ts, binning_ts>(
                std::forward<Ts>(data)...);
        } else if (!binning_ids.empty()) {

            const auto first_id{binning_ids.front()};
            binning_ids.pop();

            switch (first_id) {
                case binning::e_regular: {
                    using new_binning_ts =
                        types::push_back<binning_ts, regular_binning_t>;
                    return convert<detector_t, bounds_ts, new_binning_ts>(
                        binning_ids, std::forward<Ts>(data)...);
                }
                case binning::e_irregular: {
                    using new_binning_ts =
                        types::push_back<binning_ts, irregular_binning_t>;
                    return convert<detector_t, bounds_ts, new_binning_ts>(
                        binning_ids, std::forward<Ts>(data)...);
                }
                // Test some edge cases
                default: {
                    throw std::invalid_argument(
                        "Given type id could not be matched to a grid binning "
                        "type: " +
                        std::to_string(static_cast<std::int64_t>(first_id)));
                    break;
                }
            }
        }
    }

    /// @brief recursively build the grid: find the grid geometry (loc. coord.)
    ///
    /// @tparam bounds_ts type list that contains the bounds types
    /// @tparam binning_ts type list that contains the binning types
    ///
    /// @param grid_data grid IO payload (read from file)
    /// @param det_builder gather the grid data and build the final volume
    template <typename detector_t, typename bounds_ts, typename binning_ts,
              typename content_t>
        requires(types::size<bounds_ts> == dim) &&
        (types::size<binning_ts> == dim) static void convert(
            const std::pair<dindex, grid_payload<content_t>> &grid_data,
            detector_builder<typename detector_t::metadata, volume_builder>
                &det_builder) {

        using algebra_t = typename detector_t::algebra_type;

        // Throw expection if the accelerator link type id is invalid
        auto print_error = [](io::accel_id grid_link) {
            if (grid_link == io::accel_id::unknown) {
                throw std::invalid_argument(
                    "Unknown accelerator id in geometry file!");
            } else {
                throw std::invalid_argument(
                    "Given accelerator id could not be matched to a grid "
                    "type: " +
                    std::to_string(static_cast<std::int64_t>(grid_link)));
            }
        };

        // Need to pass actual instances to deduce contained types as
        // template parameter packs
        constexpr auto bounds = bounds_ts{};
        constexpr auto binnings = binning_ts{};

        // Check only 2-dimensional grid types
        if constexpr (dim == 2) {
            switch (grid_data.second.grid_link.type) {
                // rectangle, trapezoid, (triangle) grids
                case io::accel_id::cartesian2_grid: {
                    return convert<detector_t, cartesian2D<algebra_t>>(
                        grid_data, det_builder, bounds, binnings);
                }
                // ring/disc, annulus grids
                case io::accel_id::polar2_grid: {
                    return convert<detector_t, polar2D<algebra_t>>(
                        grid_data, det_builder, bounds, binnings);
                }
                // 2D concentric cylinder grid
                case io::accel_id::concentric_cylinder2_grid: {
                    return convert<detector_t,
                                   concentric_cylindrical2D<algebra_t>>(
                        grid_data, det_builder, bounds, binnings);
                }
                // 2D cylinder grid
                case io::accel_id::cylinder2_grid: {
                    return convert<detector_t, cylindrical2D<algebra_t>>(
                        grid_data, det_builder, bounds, binnings);
                }
                default: {
                    print_error(grid_data.second.grid_link.type);
                    break;
                }
            }
        } else if constexpr (dim == 3) {
            switch (grid_data.second.grid_link.type) {
                // cuboid grid
                case io::accel_id::cuboid3_grid: {
                    return convert<detector_t, cartesian3D<algebra_t>>(
                        grid_data, det_builder, bounds, binnings);
                }
                // 3D cylinder grid
                case io::accel_id::cylinder3_grid: {
                    return convert<detector_t, cylindrical3D<algebra_t>>(
                        grid_data, det_builder, bounds, binnings);
                }
                default: {
                    print_error(grid_data.second.grid_link.type);
                    break;
                }
            }
        } else {
            throw std::invalid_argument("No 1D grid type defined in detray");
        }
    }

    /// @brief End of recursion: build the grid from the @param grid_data
    template <typename detector_t, typename local_frame_t, typename content_t,
              typename... bounds_ts, typename... binning_ts>
        requires(sizeof...(bounds_ts) == dim) &&
        (sizeof...(binning_ts) == dim) static void convert(
            const std::pair<dindex, grid_payload<content_t>> &grid_idx_and_data,
            detector_builder<typename detector_t::metadata, volume_builder>
                &det_builder,
            types::list<bounds_ts...>, types::list<binning_ts...>) {

        using scalar_t = typename detector_t::scalar_type;

        // Assemble the grid type
        using axes_t =
            axis::multi_axis<false, local_frame_t,
                             axis::single_axis<bounds_ts, binning_ts>...>;
        using bin_t =
            std::conditional_t<bin_capacity == 0, bins::dynamic_array<value_t>,
                               bins::static_array<value_t, bin_capacity>>;
        using grid_t = grid<axes_t, bin_t, serializer_t>;

        static_assert(grid_t::dim == dim,
                      "Grid dimension does not meet dimension of grid reader");

        const auto &[sf_type, grid_data] = grid_idx_and_data;
        const auto volume_idx{
            detail::basic_converter::convert(grid_data.owner_link)};

        // Error output
        std::stringstream err_stream;
        err_stream << "Volume " << volume_idx << ": ";

        if (!det_builder.has_volume(volume_idx)) {
            err_stream << "Cannot build grid for volume "
                       << "(volume not registered in detector builder)";
            throw std::invalid_argument(err_stream.str());
        }

        // The compiler will instantiate this function for all possible types of
        // grids: Only proceed, if the grid type is known by the detector
        if constexpr (detector_t::accel::template is_defined<grid_t>() ||
                      detector_t::materials::template is_defined<grid_t>()) {

            // Decorate the current volume builder with the grid
            using builder_t = grid_builder_t<detector_t, grid_t, bin_filler_t,
                                             grid_factory_type<grid_t>>;

            auto vgr_builder =
                det_builder.template decorate<builder_t>(volume_idx);
            if (!vgr_builder) {
                throw std::runtime_error("Grid decoration failed");
            }

            // Initialize the grid axes
            std::vector<std::size_t> n_bins_per_axis{};
            std::vector<scalar_t> spans{};
            std::vector<std::vector<scalar_t>> ax_bin_edges{};

            for (const auto &axis_data : grid_data.axes) {
                n_bins_per_axis.push_back(axis_data.bins);
                std::vector<scalar_t> edges{};
                std::ranges::copy(axis_data.edges, std::back_inserter(edges));
                ax_bin_edges.emplace_back(std::move(edges));
                spans.push_back(static_cast<scalar_t>(axis_data.edges.front()));
                spans.push_back(static_cast<scalar_t>(axis_data.edges.back()));
            }

            std::vector<std::pair<typename grid_t::loc_bin_index, dindex>>
                capacities{};

            // If the grid has dynamic bin capacities, find them
            if constexpr (std::is_same_v<typename grid_t::bin_type,
                                         bins::dynamic_array<value_t>>) {
                axis::multi_bin<dim> mbin;
                for (const auto &bin_data : grid_data.bins) {
                    assert(
                        dim == bin_data.loc_index.size() &&
                        "Dimension of local bin indices in input file does not "
                        "match grid dimension");

                    // The local bin indices for the bin to be filled
                    for (const auto &[i, bin_idx] :
                         detray::views::enumerate(bin_data.loc_index)) {
                        mbin[i] = bin_idx;
                    }
                    capacities.emplace_back(mbin, bin_data.content.size());
                }
            }

            vgr_builder->set_type(sf_type);
            vgr_builder->init_grid(spans, n_bins_per_axis, capacities,
                                   ax_bin_edges);
            auto &grid = vgr_builder->get();
            const std::size_t n_bins{grid.nbins()};

            value_t entry{};
            axis::multi_bin<dim> mbin;
            for (const auto &bin_data : grid_data.bins) {

                // The local bin indices for the bin to be filled
                for (const auto &[i, bin_idx] :
                     detray::views::enumerate(bin_data.loc_index)) {
                    mbin[i] = bin_idx;
                }

                const auto gbin = grid.serializer()(grid.axes(), mbin);
                if (gbin >= n_bins) {
                    err_stream << "Bin index " << mbin << " out of bounds";
                    throw std::invalid_argument(err_stream.str());
                }

                // For now assume surfaces ids as the only grid input
                for (const auto c : bin_data.content) {
                    if (detray::detail::is_invalid_value(
                            static_cast<dindex>(c))) {
                        std::cout << "WARNING: Encountered invalid surface "
                                  << "index in grid (" << err_stream.str()
                                  << ")" << std::endl;
                        continue;
                    }
                    entry.set_volume(volume_idx);
                    entry.set_index(static_cast<dindex>(c));
                    vgr_builder->get().template populate<attach<>>(mbin, entry);
                }
            }
        } else {
            types::print<types::list<grid_t>>();
            err_stream
                << "Grid type in file does not match any grid type in detector";
            throw std::invalid_argument(err_stream.str());
        }
    }
};

}  // namespace detray::io::detail
