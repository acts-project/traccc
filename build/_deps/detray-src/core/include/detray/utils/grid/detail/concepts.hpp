/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/core/detail/container_buffers.hpp"
#include "detray/core/detail/container_views.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/grid_axis.hpp"
#include "detray/utils/concepts.hpp"

// System include(s)
#include <array>
#include <concepts>
#include <utility>

namespace detray::concepts {

template <typename A>
concept axis = requires(const A ax) {

    typename A::bounds_type;
    typename A::binning_type;
    typename A::scalar_type;

    { ax.label() }
    ->std::same_as<axis::label>;

    { ax.bounds() }
    ->std::same_as<axis::bounds>;

    { ax.binning() }
    ->std::same_as<axis::binning>;

    { ax.nbins() }
    ->std::same_as<dindex>;

    { ax.bin(typename A::scalar_type()) }
    ->std::same_as<dindex>;

    { ax.range(typename A::scalar_type(), std::array<dindex, 2>()) }
    ->std::same_as<std::array<int, 2>>;

    {
        ax.range(typename A::scalar_type(),
                 std::array<typename A::scalar_type, 2>())
    }
    ->std::same_as<std::array<int, 2>>;

    { ax.bin_edges(dindex()) }
    ->std::same_as<std::array<typename A::scalar_type, 2>>;

    { ax.bin_edges() }
    ->concepts::range_of<typename A::scalar_type>;

    { ax.span() }
    ->std::same_as<std::array<typename A::scalar_type, 2>>;

    { ax.min() }
    ->std::same_as<typename A::scalar_type>;

    { ax.max() }
    ->std::same_as<typename A::scalar_type>;
};

template <typename G>
concept grid = viewable<G>&& bufferable<G>&& requires(const G g) {

    G::dim;

    typename G::bin_type;
    typename G::value_type;
    typename G::glob_bin_index;
    typename G::loc_bin_index;
    typename G::local_frame_type;
    typename G::point_type;

    G::is_owning;

    // TODO: Implement cooridnate frame concept
    { g.get_local_frame() }
    ->std::same_as<typename G::local_frame_type>;

    { g.template get_axis<0>() }
    ->concepts::axis;

    { g.nbins() }
    ->std::same_as<dindex>;

    { g.size() }
    ->std::same_as<dindex>;

    { g.deserialize(typename G::glob_bin_index()) }
    ->std::same_as<typename G::loc_bin_index>;

    { g.serialize(typename G::loc_bin_index()) }
    ->std::same_as<typename G::glob_bin_index>;

    { g.bins() }
    ->concepts::range_of<typename G::bin_type>;

    { g.bin(typename G::glob_bin_index()) }
    ->concepts::same_as_cvref<typename G::bin_type>;

    { g.bin(typename G::loc_bin_index()) }
    ->concepts::same_as_cvref<typename G::bin_type>;

    { g.at(typename G::loc_bin_index(), dindex()) }
    ->concepts::same_as_cvref<typename G::value_type>;

    { g.at(typename G::glob_bin_index(), dindex()) }
    ->concepts::same_as_cvref<typename G::value_type>;
};

}  // namespace detray::concepts
