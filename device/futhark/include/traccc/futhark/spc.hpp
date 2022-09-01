/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"

struct futhark_context_config;
struct futhark_context;

namespace traccc::futhark {
struct spacepoint_creation
    : public algorithm<spacepoint_collection_types::host(
          const std::vector<std::pair<geometry_id, transform3>>&,
          const measurement_container_types::host&)> {
    spacepoint_creation();

    ~spacepoint_creation();

    output_type operator()(
        const std::vector<std::pair<geometry_id, transform3>>&,
        const measurement_container_types::host& cells) const;

    struct futhark_context_config* cfg;
    struct futhark_context* ctx;
};
}  // namespace traccc::futhark
