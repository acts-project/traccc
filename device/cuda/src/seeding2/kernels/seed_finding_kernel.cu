/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cooperative_groups.h>

#include <iostream>
#include <traccc/cuda/seeding2/kernels/seed_finding_kernel.hpp>
#include <traccc/cuda/seeding2/types/internal_sp.hpp>
#include <traccc/cuda/seeding2/types/kd_tree.hpp>
#include <traccc/cuda/seeding2/types/range3d.hpp>
#include <traccc/cuda/utils/definitions.hpp>
#include <traccc/cuda/utils/device_traits.hpp>
#include <traccc/cuda/utils/sort.hpp>
#include <traccc/cuda/utils/sync.hpp>
#include <traccc/edm/alt_seed.hpp>
#include <traccc/edm/internal_spacepoint.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/seeding/detail/lin_circle.hpp>
#include <traccc/seeding/doublet_finding_helper.hpp>
#include <traccc/seeding/triplet_finding_helper.hpp>

#define MAX_LOWER_SP_PER_MIDDLE 100u
#define MAX_UPPER_SP_PER_MIDDLE 100u
#define KD_TREE_TRAVERSAL_STACK_SIZE 128u
#define WARPS_PER_BLOCK 8u

/**
 * @brief Maximum difference in φ between two spacepoints adjacent in the same
 * seed.
 */
static constexpr float MAX_DELTA_PHI = 0.025f;

namespace {
struct internal_seed {
    uint32_t spacepoints[2];
    float weight;

    __host__ __device__ static internal_seed Invalid() {
        internal_seed r;

        r.weight = std::numeric_limits<float>::lowest();

        return r;
    }
};
}  // namespace

namespace traccc::cuda {
/**
 * @brief Retrieve values from a k-d tree in a given range and write them to
 * an output array.
 *
 * Performs depth-first traversal of the k-d tree, looking for candidate
 * spacepoints.
 *
 * @param[in] range 3D range in which to search.
 * @param[in] mi_idx Index of the middle spacepoint.
 * @param[in] upper True iff we're looking for upper spacepoints.
 * @param[out] output_arr Output array to write candidates to.
 * @param[out] output_cnt Total number of output candidates.
 */
__device__ void retrieve_from_tree(
    const seedfinder_config finder_conf, const seedfilter_config filter_conf,
    const internal_sp_t spacepoints, const kd_tree_t tree,
    const internal_spacepoint<spacepoint> mi, const range3d range,
    uint32_t mi_idx, bool upper, uint32_t* __restrict__ output_arr,
    uint32_t* __restrict__ output_cnt) {
    __shared__ uint32_t stack[WARPS_PER_BLOCK][KD_TREE_TRAVERSAL_STACK_SIZE];
    __shared__ uint32_t leaves[WARPS_PER_BLOCK][WARP_SIZE];

    uint32_t stack_idx = 32;

    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<WARP_SIZE> warp =
        cooperative_groups::tiled_partition<WARP_SIZE>(block);

    uint32_t idx = WARP_SIZE - 1 + warp.thread_rank();
    assert(tree[idx].type == nodetype_e::INTERNAL);
    stack[warp.meta_group_rank()][warp.thread_rank()] = idx;

    warp.sync();

    /*
     * The traversal runs until the stack is empty. Or until the entire kernel
     * explodes and crashes!
     */
    while (stack_idx > 0) {
        /*
         * We perform this traversal in parallel, so multiple threads pop a
         * value from the stack at the same time. Of course, this can only
         * happen if the thread index fits within the current size of the
         * stack.
         */
        uint32_t old_stack_idx = stack_idx;

        stack_idx -= std::min(stack_idx, warp.size());

        uint32_t current;
        bool add_children = false, add_spacepoints = false;

        if (warp.thread_rank() < old_stack_idx) {
            current = stack[warp.meta_group_rank()]
                           [old_stack_idx - (warp.thread_rank() + 1)];

            /*
             * The node must be a leaf node or an internal node, otherwise we
             * cannot (safely) traverse it.
             */
            assert(tree[current].type == nodetype_e::LEAF ||
                   tree[current].type == nodetype_e::INTERNAL);

            const range3d& crange = tree[current].range;

            /*
             * Choose the appropriate code path based on whether the node is a
             * leaf or an internal node.
             */
            add_spacepoints = tree[current].type == nodetype_e::LEAF ||
                              range.dominates(crange);
            add_children = tree[current].type == nodetype_e::INTERNAL &&
                           range.intersects(crange);
        }

        auto [vst, vsi] = warp_indexed_ballot_sync(add_spacepoints);

        if (add_spacepoints) {
            uint32_t beg = tree[current].begin, end = tree[current].end;

            for (uint32_t point_id = beg; point_id < end; point_id++) {
                /*
                 * Ensure that the point is actually valid using cuts which
                 * are more specific than those encoded in the search
                 * range.
                 */
                float phi = spacepoints[point_id].phi,
                      radius = spacepoints[point_id].radius,
                      z = spacepoints[point_id].z;

                if (range.contains(phi, radius, z) && point_id != mi_idx &&
                    doublet_finding_helper::isCompatible(
                        upper, mi,
                        internal_spacepoint<spacepoint>(
                            spacepoints[point_id].x, spacepoints[point_id].y, z,
                            radius, phi, spacepoints[point_id].link),
                        finder_conf)) {
                    /*
                     * Reserve a spot in the output for this point, and
                     * then store it if it would not exceed the output
                     * array size.
                     */
                    uint32_t idx = atomicAdd(output_cnt, 1u);

                    assert((upper && idx < MAX_UPPER_SP_PER_MIDDLE) ||
                           (!upper && idx < MAX_LOWER_SP_PER_MIDDLE));

                    output_arr[idx] = point_id;
                }
            }
        }

        auto [vct, vci] = warp_indexed_ballot_sync(add_children);

        if (add_children) {
            uint32_t idx = stack_idx + vci * 2u;
            assert(idx + 1 < KD_TREE_TRAVERSAL_STACK_SIZE);

            stack[warp.meta_group_rank()][idx] = 2 * current + 1;
            stack[warp.meta_group_rank()][idx + 1] = 2 * current + 2;
        }

        stack_idx += 2u * vct;

        /*
         * Enforce synchronization of the block between DFS steps.
         */
        warp.sync();
    }
}

/**
 * @brief Create an internal seed from three spacepoints.
 *
 * @param[in] spacepoints The array of spacepoints.
 * @param[in] lower The index of the lower spacepoint.
 * @param[in] middle The index of the middle spacepoint.
 * @param[in] upper The index of the upper spacepoint.
 *
 * @return An internal seed iff the triplet is valid, otherwise an invalid
 * seed.
 */
__device__ internal_seed make_seed(
    const seedfinder_config finder_conf, const seedfilter_config filter_conf,
    const internal_sp_t spacepoints, const kd_tree_t tree, uint32_t lower,
    uint32_t middle, internal_spacepoint<spacepoint> mi, uint32_t upper) {
    const internal_spacepoint<spacepoint> lo(
        spacepoints[lower].x, spacepoints[lower].y, spacepoints[lower].z,
        spacepoints[lower].radius, spacepoints[lower].phi,
        spacepoints[lower].link);
    const internal_spacepoint<spacepoint> hi(
        spacepoints[upper].x, spacepoints[upper].y, spacepoints[upper].z,
        spacepoints[upper].radius, spacepoints[upper].phi,
        spacepoints[upper].link);

    /*
     * Find the lin-circles for the bottom and top pair.
     */
    lin_circle lm = doublet_finding_helper::transform_coordinates<
        details::spacepoint_type::bottom>(mi, lo);
    lin_circle mh = doublet_finding_helper::transform_coordinates<
        details::spacepoint_type::top>(mi, hi);

    scalar iSinTheta2 = 1 + lm.cotTheta() * lm.cotTheta();
    scalar scatteringInRegion2 = finder_conf.maxScatteringAngle2 * iSinTheta2;
    scatteringInRegion2 *=
        finder_conf.sigmaScattering * finder_conf.sigmaScattering;
    scalar curvature, impact_parameter;

    /*
     * Borrow the compatibility code from the existing seed finding code.
     */
    if (triplet_finding_helper::isCompatible(mi, lm, mh, finder_conf,
                                             iSinTheta2, scatteringInRegion2,
                                             curvature, impact_parameter)) {
        internal_seed r;

        r.spacepoints[0] = lower;
        r.spacepoints[1] = upper;

        /*
         * Add a weight based on the impact parameter to the seed.
         */
        r.weight = -impact_parameter * filter_conf.impactWeightFactor;

        return r;
    } else {
        /*
         * If the triplet is invalid, return a bogus seed.
         */
        return internal_seed::Invalid();
    }
}

/**
 * @brief Get the basic Range3D object
 *
 * This contains only the most basic cuts, but is designed to be refined later.
 *
 * @param[in] finder_conf The seed finder configuration to use.
 *
 * @return A 3D range object.
 */
__device__ range3d get_basic_range3d(const seedfinder_config finder_conf) {
    range3d r = range3d::Infinite();

    r.r_min = 0.f;
    r.r_max = finder_conf.rMax;

    r.z_min = finder_conf.zMin;
    r.z_max = finder_conf.zMax;

    return r;
}

/**
 * @brief Return the search range for a lower spacepoint given a middle
 * spacepoint.
 *
 * @param[in] finder_conf The seed finder configuration to use.
 * @param[in] s The middle spacepoint.
 *
 * @return A three-dimensional search range.
 */
__device__ range3d get_lower_range3d(const seedfinder_config finder_conf,
                                     const internal_spacepoint<spacepoint> s) {
    range3d r = get_basic_range3d(finder_conf);

    r.r_min = std::max(r.r_min, s.radius() - finder_conf.deltaRMax);
    r.r_max = std::min(r.r_max, s.radius() - finder_conf.deltaRMin);

    float frac_r = r.r_min / s.radius();

    float z_min2 = (s.z() - finder_conf.collisionRegionMin) * frac_r +
                   finder_conf.collisionRegionMin;
    float z_max2 = (s.z() - finder_conf.collisionRegionMax) * frac_r +
                   finder_conf.collisionRegionMax;

    r.z_min = std::max(r.z_min, std::min(z_min2, s.z()));
    r.z_max = std::min(r.z_max, std::max(z_max2, s.z()));

    r.z_min = std::max(r.z_min, s.z() - 450.f);
    r.z_max = std::min(r.z_max, s.z() + 450.f);

    r.phi_min = std::max(r.phi_min, s.phi() - MAX_DELTA_PHI);
    r.phi_max = std::min(r.phi_max, s.phi() + MAX_DELTA_PHI);

    return r;
}

/**
 * @brief Return the search range for an upper spacepoint given a middle
 * spacepoint.
 *
 * @param[in] finder_conf The seed finder configuration to use.
 * @param[in] s The middle spacepoint.
 *
 * @return A three-dimensional search range.
 */
__device__ range3d get_upper_range3d(const seedfinder_config finder_conf,
                                     const internal_spacepoint<spacepoint> s) {
    range3d r = get_basic_range3d(finder_conf);

    r.r_min = std::max(r.r_min, s.radius() + finder_conf.deltaRMin);
    r.r_max = std::min(r.r_max, s.radius() + finder_conf.deltaRMax);

    float z_max2 =
        (r.r_max / s.radius()) * (s.z() - finder_conf.collisionRegionMin) +
        finder_conf.collisionRegionMin;
    float z_min2 =
        finder_conf.collisionRegionMax -
        (r.r_max / s.radius()) * (finder_conf.collisionRegionMax - s.z());

    if (s.z() > finder_conf.collisionRegionMin) {
        r.z_max = std::min(r.z_max, z_max2);
    } else if (s.z() < finder_conf.collisionRegionMax) {
        r.z_min = std::max(r.z_min, z_min2);
    }

    r.z_min = std::max(
        r.z_min, s.z() - finder_conf.cotThetaMax * (r.r_max - s.radius()));
    r.z_max = std::min(
        r.z_max, s.z() + finder_conf.cotThetaMax * (r.r_max - s.radius()));

    r.z_min = std::max(r.z_min, s.z() - 450.f);
    r.z_max = std::min(r.z_max, s.z() + 450.f);

    r.phi_min = std::max(r.phi_min, s.phi() - MAX_DELTA_PHI);
    r.phi_max = std::min(r.phi_max, s.phi() + MAX_DELTA_PHI);

    return r;
}

/**
 * @brief Seed finding helper function.
 *
 * Turns out this function actually does most of the important work.
 *
 * @param[inout] internal_seeds The array to write internal seeds to.
 * @param[in] increasing_z True iff we are looking for increasing-z seeds.
 */
__device__ void run_helper(const seedfinder_config finder_conf,
                           const seedfilter_config filter_conf,
                           const internal_sp_t spacepoints,
                           const kd_tree_t tree,
                           const internal_spacepoint<spacepoint> sp,
                           std::size_t sp_idx, internal_seed* internal_seeds,
                           bool increasing_z) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<WARP_SIZE> warp =
        cooperative_groups::tiled_partition<WARP_SIZE>(block);

    /*
     * Allocate shared memory for the lower and upper candidates.
     */
    __shared__ uint32_t lower_sps[WARPS_PER_BLOCK][MAX_LOWER_SP_PER_MIDDLE];
    __shared__ uint32_t upper_sps[WARPS_PER_BLOCK][MAX_UPPER_SP_PER_MIDDLE];

    /*
     * The leader thread initializes the candidate counts to zero.
     */
    __shared__ uint32_t num_lower[WARPS_PER_BLOCK], num_upper[WARPS_PER_BLOCK];

    if (warp.thread_rank() == 0) {
        num_lower[warp.meta_group_rank()] = 0;
        num_upper[warp.meta_group_rank()] = 0;
    }

    warp.sync();

    /*
     * Retrieve the search range for the lower spacepoints, then adjust it for
     * monotinicity in z.
     */
    range3d lower_range3d = get_lower_range3d(finder_conf, sp);

    if (increasing_z) {
        lower_range3d.z_max = std::min(lower_range3d.z_max, sp.z());
    } else {
        lower_range3d.z_min = std::max(lower_range3d.z_min, sp.z());
    }

    /*
     * Same as above, but for the upper spacepoint. Note that some of the
     * operations are reversed.
     */
    range3d upper_range3d = get_upper_range3d(finder_conf, sp);

    if (increasing_z) {
        upper_range3d.z_min = std::max(lower_range3d.z_min, sp.z());
    } else {
        upper_range3d.z_max = std::min(lower_range3d.z_max, sp.z());
    }

    /*
     * If either of the search ranges is degenerate, we can never find any
     * candidate seeds. So we can safely exit.
     */
    if (lower_range3d.phi_min >= lower_range3d.phi_max ||
        lower_range3d.r_min >= lower_range3d.r_max ||
        lower_range3d.z_min >= lower_range3d.z_max ||
        upper_range3d.phi_min >= upper_range3d.phi_max ||
        upper_range3d.r_min >= upper_range3d.r_max ||
        upper_range3d.z_min >= upper_range3d.z_max) {
        return;
    }

    /*
     * Retrieve the lower and upper spacepoint candidates from the k-d tree.
     */
    retrieve_from_tree(finder_conf, filter_conf, spacepoints, tree, sp,
                       lower_range3d, sp_idx, false,
                       lower_sps[warp.meta_group_rank()],
                       &num_lower[warp.meta_group_rank()]);

    warp.sync();

    /*
     * If we have zero lower candidates, we might as well quit; we will never
     * have any candidates!
     */
    if (num_lower[warp.meta_group_rank()] == 0) {
        return;
    }

    retrieve_from_tree(finder_conf, filter_conf, spacepoints, tree, sp,
                       upper_range3d, sp_idx, true,
                       upper_sps[warp.meta_group_rank()],
                       &num_upper[warp.meta_group_rank()]);

    warp.sync();

    /*
     * Life fast and die young! (It's an early quitting condition)
     */
    if (num_upper[warp.meta_group_rank()] == 0) {
        return;
    }

    /*
     * Calculate the total number of iterations, but take into account the
     * maximum size of the lower and upper spacepoint storage arrays.
     */
    uint32_t combinations =
        std::min(num_lower[warp.meta_group_rank()], MAX_LOWER_SP_PER_MIDDLE) *
        std::min(num_upper[warp.meta_group_rank()], MAX_UPPER_SP_PER_MIDDLE);

    /*
     * The iteration count is rounded up to a multiple of the block size to
     * ensure that none of the threads slack off and exit prematurely.
     */
    uint32_t iter_count =
        combinations + (combinations % WARP_SIZE != 0
                            ? WARP_SIZE - (combinations % WARP_SIZE)
                            : 0);

    float min_weight = std::numeric_limits<float>::lowest();

    for (uint32_t i = warp.thread_rank(); i < iter_count; i += WARP_SIZE) {
        uint32_t lower_id = i / num_upper[warp.meta_group_rank()];
        uint32_t upper_id = i % num_upper[warp.meta_group_rank()];

        /*
         * Consider the current combination and turn it into a seed. This seed
         * may be bogus! But that is okay, because the sorting step below will
         * automatically separate bogus seeds.
         */
        bool add_seed = false;
        internal_seed s;

        if (lower_id < std::min(num_lower[warp.meta_group_rank()],
                                MAX_LOWER_SP_PER_MIDDLE) &&
            upper_id < std::min(num_upper[warp.meta_group_rank()],
                                MAX_UPPER_SP_PER_MIDDLE)) {
            s = make_seed(finder_conf, filter_conf, spacepoints, tree,
                          lower_sps[warp.meta_group_rank()][lower_id], sp_idx,
                          sp, upper_sps[warp.meta_group_rank()][upper_id]);

            if (s.weight > min_weight) {
                // internal_seeds[warp.thread_rank()] = s;
                add_seed = true;
            }
        }

        auto [vt, vi] = warp_indexed_ballot_sync(add_seed);

        if (add_seed) {
            internal_seeds[WARP_SIZE - (1 + vi)] = s;
        }

        warp.sync();

        if (vt > 0) {
            /*
             * Perform odd-even sorting, making sure that the seeds with the
             * highest weight float to the top of the seed array, which is the
             * part that will be written to the global output.
             */
            warpOddEvenKeySort(
                internal_seeds, WARP_SIZE + finder_conf.maxSeedsPerSpM,
                [](const internal_seed& a, const internal_seed& b) {
                    return a.weight < b.weight;
                });

            min_weight = internal_seeds[WARP_SIZE].weight;
        }
    }
}

/**
 * @brief Main seed finding kernel.
 *
 * @note This kernel assumes exactly one warp per block and operates in a
 * one-block-per-middle-SP fashion.
 *
 */
__global__ void __launch_bounds__(WARP_SIZE* WARPS_PER_BLOCK)
    seed_finding_kernel(const seedfinder_config finder_conf,
                        const seedfilter_config filter_conf,
                        const internal_sp_t spacepoints, const kd_tree_t tree,
                        alt_seed* output_seeds, uint32_t* output_seed_size) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<WARP_SIZE> warp =
        cooperative_groups::tiled_partition<WARP_SIZE>(block);

    assert(warp.meta_group_size() == WARPS_PER_BLOCK);

    std::size_t spacepoint_id =
        block.group_index().x * WARPS_PER_BLOCK + warp.meta_group_rank();

    /*
     * The seed finding configuration defines a few criteria that allow us to
     * ignore some of the middle spacepoint candidates.
     */
    if (spacepoint_id >= spacepoints.size() ||
        spacepoints[spacepoint_id].phi < finder_conf.phiMin ||
        spacepoints[spacepoint_id].phi > finder_conf.phiMax ||
        spacepoints[spacepoint_id].radius < finder_conf.rMin ||
        spacepoints[spacepoint_id].radius > finder_conf.rMax ||
        spacepoints[spacepoint_id].z < finder_conf.zMin ||
        spacepoints[spacepoint_id].z > finder_conf.zMax) {
        return;
    }

    const internal_spacepoint<spacepoint> sp(
        spacepoints[spacepoint_id].x, spacepoints[spacepoint_id].y,
        spacepoints[spacepoint_id].z, spacepoints[spacepoint_id].radius,
        spacepoints[spacepoint_id].phi, spacepoints[spacepoint_id].link);

    /*
     * The internal seeds are stored in this shared array.
     */
    extern __shared__ internal_seed _internal_seeds[];
    internal_seed* internal_seeds =
        &_internal_seeds[warp.meta_group_rank() *
                         (WARP_SIZE + finder_conf.maxSeedsPerSpM)];

    /*
     * Initialize the internal seed array to contain only bogus seeds.
     */
    for (uint32_t i = warp.thread_rank();
         i < WARP_SIZE + finder_conf.maxSeedsPerSpM; i += WARP_SIZE) {
        internal_seeds[i].weight = std::numeric_limits<float>::lowest();
    }

    warp.sync();

    /*
     * Run the actual seed finding twice; once with monotonically increasing
     * z values, and once with monotonically decreasing ones.
     */
    run_helper(finder_conf, filter_conf, spacepoints, tree, sp, spacepoint_id,
               internal_seeds, true);
    warp.sync();
    run_helper(finder_conf, filter_conf, spacepoints, tree, sp, spacepoint_id,
               internal_seeds, false);
    warp.sync();

    /*
     * We have now completed the seeding algorithm, and we finish by writing
     * the seeds to our output array. This task is reserved for the leader
     * thread only, and all the other threads are idle. Thankfully, this should
     * not take much time at all.
     */
    if (warp.thread_rank() == 0) {
        /*
         * First, we count the number of valid seeds that we will want to write
         * to the global memory.
         */
        uint32_t num_valid = 0;

        for (uint32_t i = 0; i < finder_conf.maxSeedsPerSpM; ++i) {
            if (internal_seeds[WARP_SIZE + i].weight > -1000000.0f) {
                ++num_valid;
            }
        }

        /*
         * Next, reserve space atomically in the output array for our new
         * seeds.
         */
        uint32_t idx = atomicAdd(output_seed_size, num_valid);

        /*
         * Finally, actually write the output to global memory. Again, this is
         * done sequentially by the leader thread, but since the number of
         * seeds is so small it should be fine. Notice that we count down from
         * the end of the array, because valid seeds are always sorted higher
         * than invalid ones (if we have any).
         */
        for (uint32_t i = 0; i < num_valid; ++i) {
            alt_seed& s = output_seeds[idx + i];
            uint32_t j = WARP_SIZE + finder_conf.maxSeedsPerSpM - (i + 1);

            /*
             * Write the data back to the output.
             */
            s.spB_link = spacepoints[internal_seeds[j].spacepoints[0]].link;
            s.spM_link = spacepoints[spacepoint_id].link;
            s.spT_link = spacepoints[internal_seeds[j].spacepoints[1]].link;

            s.weight = internal_seeds[i].weight;
            s.z_vertex = 0.f;
        }
    }
}

/**
 * @brief Seeding entry point for C++ code.
 *
 * This is basically a wrapper for the seed finding kernel.
 *
 * @param[in] finder_conf The seed finder configuration to use.
 * @param[in] filter_conf The seed filter configuration to use.
 * @param[inout] mr The memory resource for allocations.
 * @param[in] spacepoints The spacepoint array.
 * @param[in] tree The k-d tree to use for searching.
 *
 * @return A unique pointer to an array of internal seeds, and the size of that
 * array.
 */
std::pair<vecmem::unique_alloc_ptr<alt_seed[]>, uint32_t> run_seeding(
    seedfinder_config finder_conf, seedfilter_config filter_conf,
    vecmem::memory_resource& mr, const internal_sp_t spacepoints,
    const kd_tree_t tree) {
    /*
     * Allocate space for output of seeds on the device.
     */
    vecmem::unique_alloc_ptr<alt_seed[]> seeds_device =
        vecmem::make_unique_alloc<alt_seed[]>(
            mr, finder_conf.maxSeedsPerSpM * spacepoints.size());

    /*
     * Allocate space for seed count on the device.
     */
    vecmem::unique_alloc_ptr<uint32_t> seed_count_device =
        vecmem::make_unique_alloc<uint32_t>(mr);
    CUDA_ERROR_CHECK(cudaMemset(seed_count_device.get(), 0, sizeof(uint32_t)));

    /*
     * Calculate the total amount of shared memory on top of that which is
     * statically defined in the CUDA kernels. This is equal to one seed for
     * each block plus one seed for each output seed.
     */
    std::size_t total_shared_memory =
        (WARPS_PER_BLOCK * (WARP_SIZE + finder_conf.maxSeedsPerSpM)) *
        sizeof(internal_seed);
    std::size_t grid_size = (spacepoints.size() / WARPS_PER_BLOCK) +
                            (spacepoints.size() % WARPS_PER_BLOCK == 0 ? 0 : 1);
    std::size_t block_size = WARPS_PER_BLOCK * WARP_SIZE;

    /*
     * Call the kernel and all that jazz.
     */
    seed_finding_kernel<<<grid_size, block_size, total_shared_memory>>>(
        finder_conf, filter_conf, spacepoints, tree, seeds_device.get(),
        seed_count_device.get());

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    /*
     * Transfer the seed count back to the host and then hand it to the user.
     */
    uint32_t seed_count_host;
    CUDA_ERROR_CHECK(cudaMemcpy(&seed_count_host, seed_count_device.get(),
                                sizeof(uint32_t), cudaMemcpyDeviceToHost));

    return {std::move(seeds_device), seed_count_host};
}
}  // namespace traccc::cuda
