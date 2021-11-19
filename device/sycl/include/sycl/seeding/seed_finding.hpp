/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <algorithm>
#include "sycl/seeding/detail/doublet_counter.hpp"          //  it's duplicate code for sycl and cuda right now
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
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/spacepoint_grid.hpp>
#include <seeding/seed_filtering.hpp>      

namespace traccc {
namespace sycl {

// Sycl seeding function object
struct seed_finding {
    /// Constructor for the sycl seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param stats_config experiment-dependent statistics estimator
    /// @param mr vecmem memory resource
    /// @param q sycl queue for kernel scheduling
    seed_finding(seedfinder_config& config,
                 std::shared_ptr<spacepoint_grid> sp_grid,
                 multiplet_estimator& estimator, 
                 vecmem::memory_resource* mr,
                 ::sycl::queue* q)
        : m_seedfinder_config(config),
          m_estimator(estimator),
          m_mr(mr),
          m_q(q),
          // initialize all vecmem containers:
          // the size of header and item vector = the number of spacepoint bins
          doublet_counter_container(sp_grid->size(false), mr),
          mid_bot_container(sp_grid->size(false), mr),
          mid_top_container(sp_grid->size(false), mr),
          triplet_counter_container(sp_grid->size(false), mr),
          triplet_container(sp_grid->size(false), mr),
          seed_container(1, mr) {

        first_alloc = true;
    }

    /// Callable operator for the sycl seed finding
    ///
    /// @return seed_container is the vecmem seed container
    host_seed_container operator()(
        host_internal_spacepoint_container& isp_container) { 

        size_t n_internal_sp = 0;

        // resize the item vectors based on the pre-estimated statistics, which
        // is experiment-dependent
        for (size_t i = 0; i < isp_container.size(); ++i) {

            // estimate the number of multiplets as a function of the middle
            // spacepoints in the bin
            size_t n_spM = isp_container.get_items()[i].size();
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

            n_internal_sp += isp_container.get_items()[i].size();
        }

        // estimate the number of seeds as a function of the internal
        // spacepoints in an event
        seed_container.get_headers()[0] = 0;
        seed_container.get_items()[0].resize(
            m_estimator.get_seeds_size(n_internal_sp));

        first_alloc = false;

        // doublet counting
        traccc::sycl::doublet_counting(m_seedfinder_config, isp_container,
                                       doublet_counter_container, m_mr, m_q);
        std::cout << doublet_counter_container.get_items().at(0)[0].n_mid_bot << " ";
        // doublet finding
        traccc::sycl::doublet_finding(m_seedfinder_config, isp_container, doublet_counter_container,
                                     mid_bot_container, mid_top_container, m_mr, m_q);

        // triplet counting
        traccc::sycl::triplet_counting(m_seedfinder_config, isp_container,
                                       doublet_counter_container,
                                       mid_bot_container, mid_top_container,
                                       triplet_counter_container, m_mr, m_q);  

        // triplet finding
        traccc::sycl::triplet_finding(m_seedfinder_config, m_seedfilter_config, isp_container,
                                      doublet_counter_container, mid_bot_container, mid_top_container,
                                      triplet_counter_container, triplet_container, m_mr, m_q);

        // weight updating
        traccc::sycl::weight_updating(m_seedfilter_config, isp_container,
                                      triplet_counter_container,
                                      triplet_container, m_mr, m_q);    

        // seed selecting
        traccc::sycl::seed_selecting(m_seedfilter_config, isp_container, doublet_counter_container,
                                     triplet_counter_container, triplet_container, seed_container, m_mr, m_q);

        return seed_container;  
        
    }
private:
    bool first_alloc;
    const seedfinder_config m_seedfinder_config;
    const seedfilter_config m_seedfilter_config;
    multiplet_estimator m_estimator;
    seed_filtering m_seed_filtering;

    host_doublet_counter_container doublet_counter_container;
    host_doublet_container mid_bot_container;
    host_doublet_container mid_top_container;
    host_triplet_counter_container triplet_counter_container;
    host_triplet_container triplet_container;
    host_seed_container seed_container;
    vecmem::memory_resource* m_mr;
    ::sycl::queue* m_q;
};

} // namespace sycl
} // namespace traccc