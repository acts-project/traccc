/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithm>
#include <cuda/seeding/detail/doublet_counter.hpp>
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
struct seed_finding : public algorithm<host_seed_container(sp_grid&&)> {

    /// Constructor for the cuda seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param mr vecmem memory resource
    seed_finding(seedfinder_config& config, unsigned int nbins,
                 vecmem::memory_resource& mr)
        : m_seedfinder_config(config),
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
    output_type operator()(sp_grid&& g2) const override {
        std::lock_guard<std::mutex> lock(*mutex);

        // reinitialize the number of multiplets to zero
        for (size_t i = 0; i < g2.nbins(); ++i) {
            doublet_counter_container.get_headers()[i].zeros();
            mid_bot_container.get_headers()[i].zeros();
            mid_top_container.get_headers()[i].zeros();
            triplet_counter_container.get_headers()[i].zeros();
            triplet_container.get_headers()[i].zeros();
        }
        seed_container.get_headers()[0] = 0;

        // resize the doublet counter container with the number of middle
        // spacepoint
        for (size_t i = 0; i < g2.nbins(); ++i) {
            size_t n_spM = g2.bin(i).size();
            doublet_counter_container.get_items()[i].resize(n_spM);
        }

        // doublet counting
        traccc::cuda::doublet_counting(m_seedfinder_config, g2,
                                       doublet_counter_container, m_mr.get());

        // resize the doublet container with the number of doublets
        for (size_t i = 0; i < g2.nbins(); ++i) {
            mid_bot_container.get_items()[i].resize(
                doublet_counter_container.get_headers()[i].n_mid_bot);
            mid_top_container.get_items()[i].resize(
                doublet_counter_container.get_headers()[i].n_mid_top);
        }

        // doublet finding
        traccc::cuda::doublet_finding(
            m_seedfinder_config, g2, doublet_counter_container,
            mid_bot_container, mid_top_container, m_mr.get());

        // resize the triplet_counter container with the number of doublets
        for (size_t i = 0; i < g2.nbins(); ++i) {
            triplet_counter_container.get_items()[i].resize(
                doublet_counter_container.get_headers()[i].n_mid_bot);
        }

        // triplet counting
        traccc::cuda::triplet_counting(m_seedfinder_config, g2,
                                       doublet_counter_container,
                                       mid_bot_container, mid_top_container,
                                       triplet_counter_container, m_mr.get());

        // resize the triplet container with the number of triplets
        for (size_t i = 0; i < g2.nbins(); ++i) {
            triplet_container.get_items()[i].resize(
                triplet_counter_container.get_headers()[i].n_triplets);
        }

        // triplet finding
        traccc::cuda::triplet_finding(
            m_seedfinder_config, m_seedfilter_config, g2,
            doublet_counter_container, mid_bot_container, mid_top_container,
            triplet_counter_container, triplet_container, m_mr.get());

        // weight updating
        traccc::cuda::weight_updating(m_seedfilter_config, g2,
                                      triplet_counter_container,
                                      triplet_container, m_mr.get());

        // resize the seed container with the number of triplets per event
        unsigned int n_triplets_per_event = 0;
        for (size_t i = 0; i < g2.nbins(); ++i) {
            n_triplets_per_event +=
                triplet_counter_container.get_headers()[i].n_triplets;
        }
        seed_container.get_items()[0].resize(n_triplets_per_event);

        // seed selecting
        traccc::cuda::seed_selecting(
            m_seedfilter_config, g2, doublet_counter_container,
            triplet_counter_container, triplet_container, seed_container,
            m_mr.get());

        return seed_container;
    }

    private:
    const seedfinder_config m_seedfinder_config;
    const seedfilter_config m_seedfilter_config;
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
