/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithm>
#include "sycl/seeding/detail/doublet_counter.hpp"         
#include "sycl/seeding/detail/multiplet_estimator.hpp"    
#include "sycl/seeding/doublet_counting.hpp"
#include "sycl/seeding/doublet_finding.hpp"
#include "sycl/seeding/seed_selecting.hpp"  
#include "sycl/seeding/triplet_counting.hpp"
#include "sycl/seeding/triplet_finding.hpp"
#include "sycl/seeding/weight_updating.hpp"     
#include <edm/internal_spacepoint.hpp>
#include <edm/seed.hpp>
#include <iostream>
#include <mutex>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/spacepoint_grid.hpp>
#include <seeding/seed_filtering.hpp>    

namespace traccc {
namespace sycl {

// Forward decleration of the kernel classes in the order of execution
class doublet_count_kernel;
class doublet_find_kernel;
class triplet_count_kernel;
class triplet_find_kernel;
class weight_update_kernel;
class seed_select_kernel;

// Sycl seeding function object
struct seed_finding : public algorithm<host_seed_container(sp_grid&&)> {

    /// Constructor for the sycl seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param stats_config experiment-dependent statistics estimator
    /// @param mr vecmem memory resource
    /// @param q sycl queue for kernel scheduling
    seed_finding(seedfinder_config& config,
                 multiplet_estimator& estimator, 
                 unsigned int nbins,
                 vecmem::memory_resource& mr,
                 ::sycl::queue* q)
        : m_seedfinder_config(config),
          m_estimator(estimator),
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
    output_type operator()(sp_grid&& g2) const override {
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
        traccc::sycl::doublet_counting(m_seedfinder_config, g2,
                                       doublet_counter_container, m_mr.get(), m_q);
        
        //doublet finding
        traccc::sycl::doublet_finding(m_seedfinder_config, g2, doublet_counter_container,
                                     mid_bot_container, mid_top_container, m_mr.get(), m_q);
        
        // triplet counting
        traccc::sycl::triplet_counting(m_seedfinder_config, g2,
                                       doublet_counter_container,
                                       mid_bot_container, mid_top_container,
                                       triplet_counter_container, m_mr.get(), m_q);  

        // triplet finding
        traccc::sycl::triplet_finding(m_seedfinder_config, m_seedfilter_config, g2,
                                      doublet_counter_container, mid_bot_container, mid_top_container,
                                      triplet_counter_container, triplet_container, m_mr.get(), m_q);

        // weight updating
        traccc::sycl::weight_updating(m_seedfilter_config, g2,
                                      triplet_counter_container,
                                      triplet_container, m_mr.get(), m_q);    
        
        // seed selecting
        traccc::sycl::seed_selecting(m_seedfilter_config, g2, doublet_counter_container,
                                     triplet_counter_container, triplet_container, seed_container, m_mr.get(), m_q);

        return seed_container;  
    }
private:
    const seedfinder_config m_seedfinder_config;
    const seedfilter_config m_seedfilter_config;
    multiplet_estimator m_estimator;
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

} // namespace sycl
} // namespace traccc