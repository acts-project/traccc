/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "seeding/seed_finding.hpp"
#include "seeding/spacepoint_grouping.hpp"

namespace traccc {

class seeding_algorithm
    : public algorithm<
          const host_spacepoint_container&,
          std::pair<host_internal_spacepoint_container, host_seed_container> > {
    public:
    struct config {};

    output_type operator()(
        const input_type& spacepoints_per_event) const override {
        output_type o;
        this->operator()(spacepoints_per_event, o);
        return o;
    }

    void operator()(const input_type& spacepoints_per_event,
                    output_type& o) const override {
        /*
        // output containers
        auto& internal_spacepoints_per_event = o.first;
        auto& seeds = o.second;

        sg(config, grid_config);
        auto internal_sp_per_event = sg(spacepoints_per_event, &resource);

        // seed finding
        traccc::seed_finding sf(config);
        auto seeds = sf(internal_sp_per_event);
        */
    }

    private:
    // algorithms
    seedfinder_config sf_cfg;
    spacepoint_grid_config grid_cfg;
    spacepoint_grouping sg;
    seed_finding sf;
};

}  // namespace traccc
