/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"

struct futhark_context_config;
struct futhark_context;

namespace traccc::futhark {

struct component_connection : algorithm<measurement_container_types::host(
                                  const cell_container_types::host& cells)> {
    component_connection();

    ~component_connection();

    output_type operator()(const cell_container_types::host& cells) const;

    struct futhark_context_config* cfg;
    struct futhark_context* ctx;
};

}  // namespace traccc::futhark
