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
};

/// Declare all seed collection types
using seed_collection_types = collection_types<seed>;

}  // namespace traccc
