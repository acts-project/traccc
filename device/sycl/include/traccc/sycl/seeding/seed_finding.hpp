/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/seed_filtering.hpp"
#include "traccc/sycl/seeding/detail/doublet_counter.hpp"
#include "traccc/sycl/seeding/doublet_counting.hpp"
#include "traccc/sycl/seeding/doublet_finding.hpp"
#include "traccc/sycl/seeding/seed_selecting.hpp"
#include "traccc/sycl/seeding/triplet_counting.hpp"
#include "traccc/sycl/seeding/triplet_finding.hpp"
#include "traccc/sycl/seeding/weight_updating.hpp"

// System include(s).
#include <algorithm>
#include <iostream>
#include <mutex>

namespace traccc {
namespace sycl {

// Sycl seeding function object
struct seed_finding : public algorithm<host_seed_container(
                          const host_spacepoint_container&, const sp_grid&)> {
    /// Constructor for the sycl seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param stats_config experiment-dependent statistics estimator
    /// @param mr vecmem memory resource
    /// @param q sycl queue for kernel scheduling
    seed_finding(seedfinder_config& config, unsigned int nbins,
                 vecmem::memory_resource& mr, ::sycl::queue* q)
        : m_seedfinder_config(config),
          m_mr(mr),
          m_q(q),
          // initialize all vecmem containers:
          // the size of header and item vector = the number of spacepoint bins
          doublet_counter_container(nbins, &m_mr.get()),
          mid_bot_container(nbins, &m_mr.get()),
          mid_top_container(nbins, &m_mr.get()),
          triplet_counter_container(nbins, &m_mr.get()),
          triplet_container(nbins, &m_mr.get()),
          seed_container(1, &m_mr.get()) {}

    /// Callable operator for the seed finding
    ///
    /// @return seed_collection is the vector of seeds per event
    output_type operator()(const host_spacepoint_container& spacepoints,
                           const sp_grid& g2) const override {
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
        traccc::sycl::doublet_counting(
            m_seedfinder_config, const_cast<sp_grid&>(g2),
            doublet_counter_container, m_mr.get(), m_q);

        // resize the doublet container with the number of doublets
        for (size_t i = 0; i < g2.nbins(); ++i) {
            mid_bot_container.get_items()[i].resize(
                doublet_counter_container.get_headers()[i].n_mid_bot);
            mid_top_container.get_items()[i].resize(
                doublet_counter_container.get_headers()[i].n_mid_top);
        }

        // doublet finding
        traccc::sycl::doublet_finding(
            m_seedfinder_config, const_cast<sp_grid&>(g2),
            doublet_counter_container, mid_bot_container, mid_top_container,
            m_mr.get(), m_q);

        // resize the triplet_counter container with the number of doublets
        for (size_t i = 0; i < g2.nbins(); ++i) {
            triplet_counter_container.get_items()[i].resize(
                doublet_counter_container.get_headers()[i].n_mid_bot);
        }

        // triplet counting
        traccc::sycl::triplet_counting(
            m_seedfinder_config, const_cast<sp_grid&>(g2),
            doublet_counter_container, mid_bot_container, mid_top_container,
            triplet_counter_container, m_mr.get(), m_q);

        // resize the triplet container with the number of triplets
        for (size_t i = 0; i < g2.nbins(); ++i) {
            triplet_container.get_items()[i].resize(
                triplet_counter_container.get_headers()[i].n_triplets);
        }

        // triplet finding
        traccc::sycl::triplet_finding(
            m_seedfinder_config, m_seedfilter_config, const_cast<sp_grid&>(g2),
            doublet_counter_container, mid_bot_container, mid_top_container,
            triplet_counter_container, triplet_container, m_mr.get(), m_q);

        // weight updating
        traccc::sycl::weight_updating(
            m_seedfilter_config, const_cast<sp_grid&>(g2),
            triplet_counter_container, triplet_container, m_mr.get(), m_q);

        // resize the seed container with the number of triplets per event
        seed_container.get_items()[0].resize(triplet_container.total_size());

        // seed selecting
        traccc::sycl::seed_selecting(
            m_seedfilter_config,
            const_cast<host_spacepoint_container&>(spacepoints),
            const_cast<sp_grid&>(g2), doublet_counter_container,
            triplet_counter_container, triplet_container, seed_container,
            m_mr.get(), m_q);

        return seed_container;
    }

    private:
    const seedfinder_config m_seedfinder_config;
    const seedfilter_config m_seedfilter_config;
    seed_filtering m_seed_filtering;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    ::sycl::queue* m_q;

    // mutable internal objects for multiplets
    mutable std::unique_ptr<std::mutex> mutex{std::make_unique<std::mutex>()};
    mutable host_doublet_counter_container doublet_counter_container;
    mutable host_doublet_container mid_bot_container;
    mutable host_doublet_container mid_top_container;
    mutable host_triplet_counter_container triplet_counter_container;
    mutable host_triplet_container triplet_container;
    mutable host_seed_container seed_container;
};

}  // namespace sycl
}  // namespace traccc
