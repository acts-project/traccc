/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/container.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc {

/// Item: seed consisting of three spacepoints, z origin and weight
struct seed {

    using link_type = typename spacepoint_container_types::host::link_type;

    link_type spB_link;
    link_type spM_link;
    link_type spT_link;

    scalar weight;
    scalar z_vertex;

    TRACCC_HOST_DEVICE
    std::array<measurement, 3> get_measurements(
        const spacepoint_container_types::const_view& spacepoints_view) const {
        const spacepoint_container_types::const_device spacepoints(
            spacepoints_view);
        return {spacepoints.at(spB_link).meas, spacepoints.at(spM_link).meas,
                spacepoints.at(spT_link).meas};
    }

    TRACCC_HOST_DEVICE
    std::array<spacepoint, 3> get_spacepoints(
        const spacepoint_container_types::const_view& spacepoints_view) const {
        const spacepoint_container_types::const_device spacepoints(
            spacepoints_view);
        return {spacepoints.at(spB_link), spacepoints.at(spM_link),
                spacepoints.at(spT_link)};
    }
};

template <typename seed_collection_t, typename spacepoint_container_t>
TRACCC_HOST std::vector<std::array<spacepoint, 3>> get_spacepoint_vector(
    const seed_collection_t& seeds, const spacepoint_container_t& container) {

    std::vector<std::array<spacepoint, 3>> result;
    result.reserve(seeds.size());

    std::transform(seeds.cbegin(), seeds.cend(), std::back_inserter(result),
                   [&](const seed& value) {
                       return value.get_spacepoints(get_data(container));
                   });

    return result;
}

/// Declare all seed collection types
using seed_collection_types = collection_types<seed>;

}  // namespace traccc
