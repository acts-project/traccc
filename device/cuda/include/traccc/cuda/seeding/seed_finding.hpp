/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/cuda/seeding/detail/doublet_counter.hpp"
#include "traccc/cuda/seeding/doublet_counting.hpp"
#include "traccc/cuda/seeding/doublet_finding.hpp"
#include "traccc/cuda/seeding/seed_selecting.hpp"
#include "traccc/cuda/seeding/triplet_counting.hpp"
#include "traccc/cuda/seeding/triplet_finding.hpp"
#include "traccc/cuda/seeding/weight_updating.hpp"
#include "traccc/edm/internal_spacepoint.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/seeding/seed_filtering.hpp"
#include "vecmem/utils/cuda/copy.hpp"

// System include(s).
#include <algorithm>

namespace traccc {
namespace cuda {

/// Seed finding for cuda
struct seed_finding : public algorithm<host_seed_collection(
                          host_spacepoint_container&&, sp_grid_buffer&&)> {

    /// Constructor for the cuda seed finding
    ///
    /// @param config is seed finder configuration parameters
    /// @param sp_grid spacepoint grid
    /// @param mr vecmem memory resource
    seed_finding(seedfinder_config& config, vecmem::memory_resource& mr)
        : m_seedfinder_config(config), m_mr(mr) {}

    /// Callable operator for the seed finding
    ///
    /// @return seed_collection is the vector of seeds per event
    output_type operator()(host_spacepoint_container&& spacepoints,
                           sp_grid_buffer&& g2_buffer) const override {
        vecmem::cuda::copy copy;
        unsigned int nbins = g2_buffer._buffer.m_size;

        // resize the doublet counter container with the number of middle
        // spacepoint
        std::vector<size_t> n_spm_per_bin;
        for (unsigned int i = 0; i < nbins; ++i) {
            n_spm_per_bin.push_back(g2_buffer._buffer.m_ptr[i].size());
        }

        doublet_counter_container_buffer dcc_buffer{
            {nbins, m_mr.get()}, {n_spm_per_bin, m_mr.get()}};
        copy.setup(dcc_buffer.headers);

        // doublet counting
        traccc::cuda::doublet_counting(m_seedfinder_config, g2_buffer,
                                       dcc_buffer, m_mr.get());

        // resize the doublet container with the number of doublets
        vecmem::vector<doublet_counter_per_bin> dcc_headers(&m_mr.get());
        copy(dcc_buffer.headers, dcc_headers);

        std::vector<size_t> n_mid_bot_per_bin;
        std::vector<size_t> n_mid_top_per_bin;
        for (const auto& h : dcc_headers) {
            n_mid_bot_per_bin.push_back(h.n_mid_bot);
            n_mid_top_per_bin.push_back(h.n_mid_top);
        }

        doublet_container_buffer mbc_buffer{{nbins, m_mr.get()},
                                            {n_mid_bot_per_bin, m_mr.get()}};
        doublet_container_buffer mtc_buffer{{nbins, m_mr.get()},
                                            {n_mid_top_per_bin, m_mr.get()}};
        copy.setup(mbc_buffer.headers);

        // doublet finding
        traccc::cuda::doublet_finding(m_seedfinder_config, dcc_headers,
                                      g2_buffer, dcc_buffer, mbc_buffer,
                                      mtc_buffer, m_mr.get());

        vecmem::vector<doublet_per_bin> mbc_headers(&m_mr.get());
        copy(mbc_buffer.headers, mbc_headers);

        triplet_counter_container_buffer tcc_buffer{
            {nbins, m_mr.get()}, {n_mid_bot_per_bin, m_mr.get()}};
        copy.setup(tcc_buffer.headers);

        // triplet counting
        traccc::cuda::triplet_counting(m_seedfinder_config, mbc_headers,
                                       g2_buffer, dcc_buffer, mbc_buffer,
                                       mtc_buffer, tcc_buffer, m_mr.get());

        vecmem::vector<triplet_counter_per_bin> tcc_headers(&m_mr.get());
        copy(tcc_buffer.headers, tcc_headers);

        std::vector<size_t> n_triplets_per_bin;
        for (const auto& h : tcc_headers) {
            n_triplets_per_bin.push_back(h.n_triplets);
        }

        triplet_container_buffer tc_buffer{{nbins, m_mr.get()},
                                           {n_triplets_per_bin, m_mr.get()}};
        copy.setup(tc_buffer.headers);

        // triplet finding
        traccc::cuda::triplet_finding(m_seedfinder_config, m_seedfilter_config,
                                      tcc_headers, g2_buffer, dcc_buffer,
                                      mbc_buffer, mtc_buffer, tcc_buffer,
                                      tc_buffer, m_mr.get());
        vecmem::vector<triplet_per_bin> tc_headers(&m_mr.get());
        copy(tc_buffer.headers, tc_headers);

        traccc::cuda::weight_updating(m_seedfilter_config, tc_headers,
                                      g2_buffer, tcc_buffer, tc_buffer,
                                      m_mr.get());

        auto n_triplets = std::accumulate(n_triplets_per_bin.begin(),
                                          n_triplets_per_bin.end(), 0);
        vecmem::data::vector_buffer<seed> seed_buffer(n_triplets, 0,
                                                      m_mr.get());
        copy.setup(seed_buffer);

        // seed selecting
        traccc::cuda::seed_selecting(
            m_seedfilter_config, dcc_headers, spacepoints, g2_buffer,
            dcc_buffer, tcc_buffer, tc_buffer, seed_buffer, m_mr.get());

        host_seed_collection seed_collection(&m_mr.get());
        copy(seed_buffer, seed_collection);

        return seed_collection;
    }

    private:
    const seedfinder_config m_seedfinder_config;
    const seedfilter_config m_seedfilter_config;
    seed_filtering m_seed_filtering;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace cuda
}  // namespace traccc
