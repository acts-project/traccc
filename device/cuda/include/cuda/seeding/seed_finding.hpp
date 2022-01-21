/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithm>
#include <cuda/seeding/detail/doublet_counter.hpp>
#include <cuda/seeding/detail/multiplet_estimator.hpp>
#include <cuda/seeding/doublet_counting.hpp>
#include <cuda/seeding/doublet_finding.hpp>
#include <cuda/seeding/seed_selecting.hpp>
#include <cuda/seeding/triplet_counting.hpp>
#include <cuda/seeding/triplet_finding.hpp>
#include <cuda/seeding/weight_updating.hpp>
#include <edm/internal_spacepoint.hpp>
#include <edm/seed.hpp>
#include <iostream>
#include <mutex>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/spacepoint_grid.hpp>
#include <seeding/seed_filtering.hpp>

namespace traccc {
namespace cuda {

/// Seed finding for cuda
struct seed_finding : public algorithm<host_seed_container(
                          host_spacepoint_container&&, sp_grid&&)> {

    /// Constructor for the cuda seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param stats_config experiment-dependent statistics estimator
    /// @param mr vecmem memory resource
    seed_finding(seedfinder_config& config, multiplet_estimator& estimator,
                 unsigned int nbins, vecmem::memory_resource& mr)
        : m_seedfinder_config(config),
          m_estimator(estimator),
          m_mr(mr),
          doublet_counter_container(nbins, &m_mr.get()),
          mid_bot_container(nbins, &m_mr.get()),
          mid_top_container(nbins, &m_mr.get()),
          triplet_counter_container(nbins, &m_mr.get()),
          triplet_container(nbins, &m_mr.get()),
          seed_container(1, &m_mr.get()) {}

    /// Callable operator for the seed finding
    ///
    /// @return seed_collection is the vector of seeds per event
    output_type operator()(host_spacepoint_container&& spacepoints,
                           sp_grid&& g2) const override {
        std::lock_guard<std::mutex> lock(*mutex);

        size_t n_internal_sp = 0;

        // resize the item vectors based on the pre-estimated statistics, which
        // is experiment-dependent
        for (size_t i = 0; i < g2.nbins(); ++i) {

            // estimate the number of multiplets as a function of the middle
            // spacepoints in the bin
            size_t n_spM = g2.bin(i).size();
            size_t n_mid_bot_doublets =
                m_estimator.get_mid_bot_doublets_size(n_spM);
            size_t n_mid_top_doublets =
                m_estimator.get_mid_top_doublets_size(n_spM);
            size_t n_triplets = m_estimator.get_triplets_size(n_spM);

            // zero initialization
            doublet_counter_container.get_headers()[i] = 0;
            mid_bot_container.get_headers()[i] = 0;
            mid_top_container.get_headers()[i] = 0;
            triplet_counter_container.get_headers()[i] = 0;
            triplet_container.get_headers()[i] = 0;

            // resize the item vectors in container
            doublet_counter_container.get_items()[i].resize(n_spM);
            mid_bot_container.get_items()[i].resize(n_mid_bot_doublets);
            mid_top_container.get_items()[i].resize(n_mid_top_doublets);
            triplet_counter_container.get_items()[i].resize(n_mid_bot_doublets);
            triplet_container.get_items()[i].resize(n_triplets);

            n_internal_sp += n_spM;
        }

        // estimate the number of seeds as a function of the internal
        // spacepoints in an event
        seed_container.get_headers()[0] = 0;
        seed_container.get_items()[0].resize(
            m_estimator.get_seeds_size(n_internal_sp));

        // doublet counting
        traccc::cuda::doublet_counting(m_seedfinder_config, g2,
                                       doublet_counter_container, m_mr.get());

        // doublet finding
        traccc::cuda::doublet_finding(
            m_seedfinder_config, g2, doublet_counter_container,
            mid_bot_container, mid_top_container, m_mr.get());

        // triplet counting
        traccc::cuda::triplet_counting(m_seedfinder_config, g2,
                                       doublet_counter_container,
                                       mid_bot_container, mid_top_container,
                                       triplet_counter_container, m_mr.get());

        // triplet finding
        traccc::cuda::triplet_finding(
            m_seedfinder_config, m_seedfilter_config, g2,
            doublet_counter_container, mid_bot_container, mid_top_container,
            triplet_counter_container, triplet_container, m_mr.get());

        // weight updating
        traccc::cuda::weight_updating(m_seedfilter_config, g2,
                                      triplet_counter_container,
                                      triplet_container, m_mr.get());

        // seed selecting
        traccc::cuda::seed_selecting(
            m_seedfilter_config, spacepoints, g2, doublet_counter_container,
            triplet_counter_container, triplet_container, seed_container,
            m_mr.get());
        return seed_container;
    }

    private:
    const seedfinder_config m_seedfinder_config;
    const seedfilter_config m_seedfilter_config;
    multiplet_estimator m_estimator;
    seed_filtering m_seed_filtering;
    std::reference_wrapper<vecmem::memory_resource> m_mr;

    // mutable internal objects for multiplets
    mutable std::unique_ptr<std::mutex> mutex{std::make_unique<std::mutex>()};
    mutable host_doublet_counter_container doublet_counter_container;
    mutable host_doublet_container mid_bot_container;
    mutable host_doublet_container mid_top_container;
    mutable host_triplet_counter_container triplet_counter_container;
    mutable host_triplet_container triplet_container;
    mutable host_seed_container seed_container;
};

}  // namespace cuda
}  // namespace traccc
