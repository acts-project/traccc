/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithm>
#include <cuda/seeding/detail/doublet_counter.hpp>
#include <cuda/seeding/detail/stats_config.hpp>
#include <cuda/seeding/doublet_counting.cuh>
#include <cuda/seeding/doublet_finding.cuh>
#include <cuda/seeding/seed_selecting.cuh>
#include <cuda/seeding/triplet_counting.cuh>
#include <cuda/seeding/triplet_finding.cuh>
#include <cuda/seeding/weight_updating.cuh>
#include <edm/internal_spacepoint.hpp>
#include <edm/seed.hpp>
#include <iostream>
#include <seeding/detail/seeding_config.hpp>

namespace traccc {
namespace cuda {

/// Seed finding for cuda
struct seed_finding {
    /// Constructor for the cuda seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param stats_config experiment-dependent statistics estimator
    /// @param mr vecmem memory resource
    seed_finding(seedfinder_config& config,
                 std::shared_ptr<spacepoint_grid> sp_grid,
                 stats_config* stats_cfg, vecmem::memory_resource* mr)
        : m_seedfinder_config(config),
          m_sp_grid(sp_grid),
          m_stats_config(stats_cfg),
          m_mr(mr),

          // initialize all vecmem containers:
          // the size of header and item vector = the number of spacepoint bins
          doublet_counter_container(
              {host_doublet_counter_container::header_vector(
                   sp_grid->size(false), 0, mr),
               host_doublet_counter_container::item_vector(sp_grid->size(false),
                                                           mr)}),

          mid_bot_container(
              {host_doublet_container::header_vector(sp_grid->size(false), 0,
                                                     mr),
               host_doublet_container::item_vector(sp_grid->size(false), mr)}),

          mid_top_container(
              {host_doublet_container::header_vector(sp_grid->size(false), 0,
                                                     mr),
               host_doublet_container::item_vector(sp_grid->size(false), mr)}),

          triplet_counter_container(
              {host_triplet_counter_container::header_vector(
                   sp_grid->size(false), 0, mr),
               host_triplet_counter_container::item_vector(sp_grid->size(false),
                                                           mr)}),

          triplet_container(
              {host_triplet_container::header_vector(sp_grid->size(false), 0,
                                                     mr),
               host_triplet_container::item_vector(sp_grid->size(false), mr)}),
          seed_container({host_seed_container::header_vector(1, 0, mr),
                          host_seed_container::item_vector(1, mr)}) {
        first_alloc = true;
    }

    /// Callable operator for the cuda seed finding
    ///
    /// @return seed_container is the vecmem seed container
    host_seed_container operator()(
        host_internal_spacepoint_container& isp_container) {

        size_t n_internal_sp = 0;

        // resize the item vectors based on the pre-estimated statistics, which
        // is experiment-dependent
        for (size_t i = 0; i < isp_container.headers.size(); ++i) {

            // estimate the number of multiplets as a function of the middle
            // spacepoints in the bin
            size_t n_spM = isp_container.items[i].size();
            size_t n_mid_bot_doublets =
                m_stats_config->get_mid_bot_doublets_size(n_spM);
            size_t n_mid_top_doublets =
                m_stats_config->get_mid_top_doublets_size(n_spM);
            size_t n_triplets = m_stats_config->get_triplets_size(n_spM);

            // zero initialization
            doublet_counter_container.headers[i] = 0;
            mid_bot_container.headers[i] = 0;
            mid_top_container.headers[i] = 0;
            triplet_counter_container.headers[i] = 0;
            triplet_container.headers[i] = 0;

            // resize the item vectors in container
            doublet_counter_container.items[i].resize(n_spM);
            mid_bot_container.items[i].resize(n_mid_bot_doublets);
            mid_top_container.items[i].resize(n_mid_top_doublets);
            triplet_counter_container.items[i].resize(n_mid_bot_doublets);
            triplet_container.items[i].resize(n_triplets);

            n_internal_sp += isp_container.items[i].size();
        }

        // estimate the number of seeds as a function of the internal
        // spacepoints in an event
        seed_container.headers[0] = 0;
        seed_container.items[0].resize(
            m_stats_config->get_seeds_size(n_internal_sp));

        first_alloc = false;

        // doublet counting
        traccc::cuda::doublet_counting(m_seedfinder_config, isp_container,
                                       doublet_counter_container, m_mr);

        // doublet finding
        traccc::cuda::doublet_finding(
            m_seedfinder_config, isp_container, doublet_counter_container,
            mid_bot_container, mid_top_container, m_mr);

        // triplet counting
        traccc::cuda::triplet_counting(m_seedfinder_config, isp_container,
                                       doublet_counter_container,
                                       mid_bot_container, mid_top_container,
                                       triplet_counter_container, m_mr);

        // triplet finding
        traccc::cuda::triplet_finding(
            m_seedfinder_config, m_seedfilter_config, isp_container,
            doublet_counter_container, mid_bot_container, mid_top_container,
            triplet_counter_container, triplet_container, m_mr);

        // weight updating
        traccc::cuda::weight_updating(m_seedfilter_config, isp_container,
                                      triplet_counter_container,
                                      triplet_container, m_mr);

        // seed selecting
        traccc::cuda::seed_selecting(
            m_seedfilter_config, isp_container, doublet_counter_container,
            triplet_counter_container, triplet_container, seed_container, m_mr);

        return seed_container;
    }

    private:
    bool first_alloc;
    const seedfinder_config m_seedfinder_config;
    const seedfilter_config m_seedfilter_config;
    std::shared_ptr<spacepoint_grid> m_sp_grid;
    stats_config* m_stats_config;
    seed_filtering m_seed_filtering;

    host_doublet_counter_container doublet_counter_container;
    host_doublet_container mid_bot_container;
    host_doublet_container mid_top_container;
    host_triplet_counter_container triplet_counter_container;
    host_triplet_container triplet_container;
    host_seed_container seed_container;
    vecmem::memory_resource* m_mr;
};

}  // namespace cuda
}  // namespace traccc
