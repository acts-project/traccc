/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void fill_path_store(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const fill_path_store_shared_payload& shared,
    collection_types<int2>::view d_path_store_view,
    const collection_types<unsigned int>::const_view& d_output_graph_view,
    const collection_types<unsigned char>::const_view& d_levels_view,
    unsigned int& nPathStoreSizeCounter, const unsigned int nTerminus,
    const unsigned int nTerminusPerBlock, const unsigned int max_num_neighbours,
    const unsigned int nReachablePaths) {

    const unsigned int threadIndex = thread_id.getLocalThreadIdX();
    const unsigned int blockIndex = thread_id.getBlockIdX();
    const unsigned int blockSize = thread_id.getBlockDimX();

    collection_types<int2>::device d_path_store(d_path_store_view);
    const collection_types<unsigned int>::const_device d_output_graph(
        d_output_graph_view);
    const collection_types<unsigned char>::const_device d_levels(d_levels_view);

    if (threadIndex == 0) {
        shared.n_live_paths = 0;
    }
    barrier.blockBarrier();

    // Row-major output graph: each edge owns a contiguous block of
    // edge_size = 2 + 1 + max_num_neighbours ints.
    const unsigned int edge_size = 2u + 1u + max_num_neighbours;
    unsigned int path_idx = threadIndex + blockIndex * nTerminusPerBlock;

    if (threadIndex < nTerminusPerBlock && path_idx < nTerminus) {
        const int2 path = d_path_store[path_idx];
        const unsigned int edge_pos =
            edge_size * static_cast<unsigned int>(path.x);
        const unsigned int nNei = d_output_graph[edge_pos + gbts_consts::nNei];
        const unsigned char level = d_levels[static_cast<unsigned int>(path.x)];
        for (unsigned int nei = 0; nei < nNei; ++nei) {
            const unsigned int edge_idx =
                d_output_graph[edge_pos + gbts_consts::nei_start + nei];
            if (level != d_levels[static_cast<unsigned int>(edge_idx)] + 1) {
                continue;
            }
            const unsigned int live_idx = static_cast<unsigned int>(
                vecmem::device_atomic_ref<int>(shared.n_live_paths)
                    .fetch_add(1));
            if (live_idx >=
                static_cast<unsigned int>(
                    traccc::device::gbts_consts::live_path_buffer)) {
                break;
            }
            const unsigned int new_path_idx =
                vecmem::device_atomic_ref<unsigned int>(nPathStoreSizeCounter)
                    .fetch_add(1u);
            d_path_store[new_path_idx] = make_int2(static_cast<int>(edge_idx),
                                                   static_cast<int>(path_idx));
            shared.live_paths[static_cast<unsigned int>(live_idx)] =
                make_uint2(edge_idx, new_path_idx);
        }
    }
    barrier.blockBarrier();

    traccc::uint2 path = make_uint2(0u, 0u);
    bool has_path = false;

    while (shared.n_live_paths > 0) {
        has_path = false;
        if (threadIndex == 0) {
            const int buf_size =
                static_cast<int>(traccc::device::gbts_consts::live_path_buffer);
            shared.n_live_paths = (shared.n_live_paths < buf_size)
                                      ? shared.n_live_paths
                                      : buf_size;
        }
        barrier.blockBarrier();
        if (static_cast<int>(threadIndex) < shared.n_live_paths) {
            path = shared.live_paths[static_cast<unsigned int>(
                shared.n_live_paths - static_cast<int>(threadIndex) - 1)];
            has_path = true;
        }
        barrier.blockBarrier();
        if (threadIndex == 0) {
            shared.n_live_paths =
                (shared.n_live_paths < static_cast<int>(blockSize))
                    ? 0
                    : shared.n_live_paths - static_cast<int>(blockSize);
        }
        barrier.blockBarrier();
        if (has_path) {
            const unsigned int edge_pos = edge_size * path.x;
            const unsigned int nNei =
                d_output_graph[edge_pos + gbts_consts::nNei];
            const unsigned char level = d_levels[path.x];
            for (unsigned int nei = 0; nei < nNei; ++nei) {
                const unsigned int edge_idx =
                    d_output_graph[edge_pos + gbts_consts::nei_start + nei];
                if (level != d_levels[edge_idx] + 1) {
                    continue;
                }
                path_idx = vecmem::device_atomic_ref<unsigned int>(
                               nPathStoreSizeCounter)
                               .fetch_add(1u);
                if (path_idx >= nReachablePaths) {
                    break;
                }
                const int live_idx =
                    vecmem::device_atomic_ref<int>(shared.n_live_paths)
                        .fetch_add(1);
                if (live_idx >=
                    static_cast<int>(
                        traccc::device::gbts_consts::live_path_buffer)) {
                    break;
                }
                d_path_store[path_idx] = make_int2(static_cast<int>(edge_idx),
                                                   static_cast<int>(path.y));
                shared.live_paths[static_cast<unsigned int>(live_idx)] =
                    make_uint2(edge_idx, path_idx);
            }
        }
        barrier.blockBarrier();
    }
}

}  // namespace traccc::device
