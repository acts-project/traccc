/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// SYCL library include(s).
#include "traccc/sycl/seeding/seed_finding.hpp"

// SYCL library include(s).
#include "doublet_counter.hpp"
#include "doublet_counting.hpp"
#include "doublet_finding.hpp"
#include "seed_selecting.hpp"
#include "triplet_counter.hpp"
#include "triplet_counting.hpp"
#include "triplet_finding.hpp"
#include "weight_updating.hpp"

// Project include(s).
#include "traccc/seeding/common/doublet.hpp"
#include "traccc/seeding/common/triplet.hpp"

// VecMem include(s).
#include "vecmem/utils/sycl/copy.hpp"

namespace traccc::sycl {

seed_finding::seed_finding(const seedfinder_config& config,
                           vecmem::memory_resource& mr, queue_wrapper queue)
    : m_seedfinder_config(config), m_mr(mr), m_queue(queue) {}

seed_finding::output_type seed_finding::operator()(
    const host_spacepoint_container& spacepoints, const sp_grid& g2) const {

    // The number of bins.
    const std::size_t nbins = g2.nbins();

    // create the doublet counter container
    host_doublet_counter_container doublet_counter_container(nbins,
                                                             &m_mr.get());
    for (std::size_t i = 0; i < nbins; ++i) {
        doublet_counter_container.get_headers()[i].zeros();
        doublet_counter_container.get_items()[i].resize(g2.bin(i).size());
    }

    // doublet counting
    doublet_counting(m_seedfinder_config, const_cast<sp_grid&>(g2),
                     doublet_counter_container, m_mr.get(), m_queue);

    // create the doublet container with the number of doublets
    host_doublet_container mid_bot_container(nbins, &m_mr.get());
    host_doublet_container mid_top_container(nbins, &m_mr.get());
    for (std::size_t i = 0; i < nbins; ++i) {
        mid_bot_container.get_headers()[i].zeros();
        mid_bot_container.get_items()[i].resize(
            doublet_counter_container.get_headers()[i].n_mid_bot);
        mid_top_container.get_headers()[i].zeros();
        mid_top_container.get_items()[i].resize(
            doublet_counter_container.get_headers()[i].n_mid_top);
    }

    // doublet finding
    doublet_finding(m_seedfinder_config, const_cast<sp_grid&>(g2),
                    doublet_counter_container, mid_bot_container,
                    mid_top_container, m_mr.get(), m_queue);

    // create the triplet_counter container with the number of doublets
    host_triplet_counter_container triplet_counter_container(nbins,
                                                             &m_mr.get());
    for (std::size_t i = 0; i < nbins; ++i) {
        triplet_counter_container.get_headers()[i].zeros();
        triplet_counter_container.get_items()[i].resize(
            doublet_counter_container.get_headers()[i].n_mid_bot);
    }

    // triplet counting
    triplet_counting(m_seedfinder_config, const_cast<sp_grid&>(g2),
                     doublet_counter_container, mid_bot_container,
                     mid_top_container, triplet_counter_container, m_mr.get(),
                     m_queue);

    // create the triplet container with the number of triplets
    host_triplet_container triplet_container(nbins, &m_mr.get());
    for (size_t i = 0; i < nbins; ++i) {
        triplet_container.get_headers()[i].zeros();
        triplet_container.get_items()[i].resize(
            triplet_counter_container.get_headers()[i].n_triplets);
    }

    // triplet finding
    triplet_finding(
        m_seedfinder_config, m_seedfilter_config, const_cast<sp_grid&>(g2),
        doublet_counter_container, mid_bot_container, mid_top_container,
        triplet_counter_container, triplet_container, m_mr.get(), m_queue);

    // weight updating
    weight_updating(m_seedfilter_config, const_cast<sp_grid&>(g2),
                    triplet_counter_container, triplet_container, m_mr.get(),
                    m_queue);

    vecmem::sycl::copy copy{m_queue.queue()};
    vecmem::data::vector_buffer<seed> seed_buffer(
        triplet_container.total_size(), 0, m_mr.get());
    copy.setup(seed_buffer);

    // seed selecting
    seed_selecting(m_seedfilter_config,
                   const_cast<host_spacepoint_container&>(spacepoints),
                   const_cast<sp_grid&>(g2), doublet_counter_container,
                   triplet_counter_container, triplet_container, seed_buffer,
                   m_mr.get(), m_queue);

    host_seed_collection seed_collection(&m_mr.get());
    copy(seed_buffer, seed_collection);

    return seed_collection;
}

}  // namespace traccc::sycl
