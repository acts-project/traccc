/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cooperative_groups.h>

#include <iostream>
#include <limits>
#include <memory>
#include <traccc/cuda/seeding2/kernels/seed_finding_kernel.hpp>
#include <traccc/cuda/seeding2/seed_finding.hpp>
#include <traccc/cuda/seeding2/types/kd_tree.hpp>
#include <traccc/cuda/seeding2/types/range3d.hpp>
#include <traccc/cuda/utils/definitions.hpp>
#include <traccc/cuda/utils/device_traits.hpp>
#include <traccc/cuda/utils/sort.hpp>
#include <traccc/edm/seed.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/seeding/detail/lin_circle.hpp>
#include <traccc/seeding/detail/seeding_config.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vector>

namespace traccc::cuda {
/**
 * @brief Bit mask for halo point indices.
 *
 * Our phi-axis search range wraps around, and k-d trees do not natively
 * support this. Thankfully, we can solve this problem by simply duplicating
 * points at the edges of the phi range. These are called phi halo points. We
 * mark these points by setting the MSB in their index. This means that, in
 * principle, the index is invalid, but it also saves quite a lot of space.
 * This mask contains the exact bit that is set as the mask.
 */
constexpr uint32_t AUXILIARY_INDEX_MASK = (1u << 31);

/**
 * @brief Width of phi-halo range.
 *
 * This value defines the range in which points are copied and treated as halo
 * points. If this width is ε, then points below -π + ε and above π - ε are
 * halo points. This value should be at least as large as the maximum phi delta
 * betweel spacepoints.
 */
// constexpr float PHI_HALO_WIDTH = 0.06f;

/**
 * @brief Get coordinate value for a halo-aware index in a given list of
 * spacepoints.
 *
 * For non-halo points, this function simply returns the phi value of the
 * point. But for halo points, it calculates the proper adjusted phi value. For
 * points in the range -π ≤ φ ≤ -π + ε, this is φ + 2π. For points in the range
 * π - ε ≤ φ ≤ π it is φ - 2π.
 *
 * @param[in] spacepoints The array of spacepoints.
 * @param[in] i The halo-aware index in the spacepoint array.
 *
 * @return The ϕ value of the spacepoint with the given index.
 */
__device__ float get_coordinate(const internal_sp_t spacepoints, uint32_t i,
                                pivot_e dim) {
    bool is_real = !(i & AUXILIARY_INDEX_MASK);

    if (is_real) {
        if (dim == pivot_e::Phi) {
            return spacepoints[i].phi;
        } else if (dim == pivot_e::R) {
            return spacepoints[i].radius;
        } else {
            return spacepoints[i].z;
        }
    } else {
        uint32_t real_i = i & ~AUXILIARY_INDEX_MASK;

        if (dim == pivot_e::Phi) {
            float phi = spacepoints[real_i].phi;

            if (phi >= 0) {
                return phi - 2.f * static_cast<float>(M_PI);
            } else {
                return phi + 2.f * static_cast<float>(M_PI);
            }
        } else if (dim == pivot_e::R) {
            return spacepoints[real_i].radius;
        } else {
            return spacepoints[real_i].z;
        }
    }
}

/**
 * @brief Initialize the index vector used to build the k-d tree.
 *
 * Our k-d tree construction algorithm keeps track of spacepoints using a big
 * index array, which is initialized here. Given n spacepoints, this has n + m
 * elements, where m is the number of halo points.
 *
 * @param[in] spacepoints The list of spacepoints.
 * @param[out] indices The output array to write to.
 * @param[in] sps The total number of spacepoints.
 * @param[out] extra_indices The output halo point count.
 */
__global__ void initialize_index_vector(internal_sp_t spacepoints,
                                        uint32_t* __restrict__ indices,
                                        uint32_t sps,
                                        uint32_t* __restrict__ extra_indices) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    if (grid.thread_rank() < sps) {
        indices[grid.thread_rank()] = grid.thread_rank();

        /*
         * If we are a halo point, add an extra index for our evil twin!
         */
        // if (spacepoints.phi(grid.thread_rank()) < (-M_PI + PHI_HALO_WIDTH) ||
        //     spacepoints.phi(grid.thread_rank()) > (M_PI - PHI_HALO_WIDTH)) {
        //     indices[sps + atomicAdd(extra_indices, 1u)] =
        //         grid.thread_rank() | AUXILIARY_INDEX_MASK;
        // }
    }
}

/**
 * @brief Kernel that initializes the k-d tree.
 *
 * This kernel simply sets every node in a given k-d tree to be non-extant
 * except for the root (at index 0) which is considered an incomplete node.
 *
 * @param[out] tree The array of tree nodes to write to.
 * @param[in] num_nodes The total number of tree nodes.
 * @param[in] num_indices The total number of indices.
 */
__global__ void initialize_kd_tree(kd_tree_t tree, uint32_t num_nodes,
                                   uint32_t num_indices,
                                   uint32_t* __restrict__ work_list,
                                   uint32_t* __restrict__ work_list_size) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    if (grid.thread_rank() == 0) {
        /*
         * The root node starts as an incomplete node covering all indices in
         * the index array. Its range is initialized as an infinite range.
         */
        tree[0].type = nodetype_e::LEAF;
        tree[0].begin = 0;
        tree[0].end = num_indices;
        tree[0].range = range3d::Infinite();

        work_list[atomicAdd(work_list_size, 1u)] = 0;
    } else if (grid.thread_rank() < num_nodes) {
        /*
         * All other nodes are non-extant.
         */
        tree[grid.thread_rank()].type = nodetype_e::NON_EXTANT;
    }
}

/**
 * @brief Helper function to determine the next pivot axis.
 *
 * k-d trees work by pivoting along a certain axis. In this implementation, we
 * simply cycle through dimensions because that's very fast. We go in the order
 * ϕ to r to z, and then repeat.
 *
 * @param[in] p The old pivot axis.
 *
 * @return The new pivot axis.
 */
__device__ pivot_e next_pivot(pivot_e p) {
    if (p == pivot_e::Phi) {
        return pivot_e::R;
    } else if (p == pivot_e::R) {
        return pivot_e::Z;
    } else {
        return pivot_e::Phi;
    }
}

/**
 * @brief Approximate mean finding algorithm for large numbers of spacepoints.
 *
 * When we have a lot of spacepoints to find the median for, doing so exactly
 * is too slow. Thus, we use a sampling approach; we select a subset of points
 * and calculate the median of those points.
 *
 * @param[in] spacepoints The list of spacepoints.
 * @param[inout] indices The list of indices for _this_ computation.
 * @param[in] num_indices The number of indices for _this_ function call.
 * @param[in] pivot The requested axis along which to split.
 *
 * @return The approximate median value of the given indices in the given
 * pivot axis.
 */
__device__ float get_approximate_median(internal_sp_t spacepoints,
                                        const uint32_t* __restrict__ indices,
                                        uint32_t num_indices, pivot_e pivot) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();

    /*
     * First, we declare some shared memory which will hold our samples from
     * which to compute the median. We assume a maximum of one item per thread.
     */
    extern __shared__ float values[];

    /*
     * Assert that the number of indices is at least as big as the block size,
     * otherwise this computation might not work as designed!
     */
    assert(num_indices >= block.size());

    /*
     * Calculate the stride between samples in the spacepoint array.
     */
    uint32_t delta = num_indices / block.size();

    /*
     * Calculate the index for the current thread.
     */
    uint32_t index = indices[(block.thread_rank() * delta) % num_indices];

    /*
     * Calculate the value of the chosen spacepoint in the requested pivot
     * dimension and store it in the shared memory.
     */
    values[block.thread_rank()] = get_coordinate(spacepoints, index, pivot);

    blockOddEvenKeySort(values, static_cast<uint32_t>(block.size()),
                        std::less<float>());

    /*
     * Now that our sorting is complete, we can simply return the middle value
     * in the range to get the median.
     */
    return values[(block.size() + 1) / 2];
}

/**
 * @brief Exact mean finding algorithm for small ranges.
 *
 * When the array of indices is relatively small, we can afford to find the
 * exact median. We do this by sorting the data and then returning the index
 * (rather than the value) of the median.
 *
 * @warning This function has the side effect that it leaves the index array
 * sorted!
 *
 * @param[in] spacepoints The array of spacepoints.
 * @param[inout] indices The array of indices for this function call.
 * @param[in] num_indices The number of indices for this function call.
 * @param[in] pivot The requested pivot axis.
 */
__device__ uint32_t sort_and_get_median(internal_sp_t spacepoints,
                                        const uint32_t* __restrict__ indices,
                                        uint32_t* __restrict__ index_buffer,
                                        uint32_t num_indices, pivot_e pivot) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();

    for (std::size_t i = block.thread_index().x; i < num_indices;
         i += block.size()) {
        index_buffer[i] = indices[i];
    }

    /*
     * Simple perform an odd-even sort to sort the range of indices by the
     * value of the corresponding SPs in the requested pivot axis.
     */
    blockOddEvenKeySort(
        index_buffer, num_indices,
        [&spacepoints, pivot](const uint32_t& a, const uint32_t& b) {
            return get_coordinate(spacepoints, a, pivot) <
                   get_coordinate(spacepoints, b, pivot);
        });

    /*
     * Remember, we are returning an index here, not a value!
     */
    return (num_indices + 1) / 2;
}

/**
 * @brief Create an internal node in the k-d tree.
 *
 * This function constructs an internal node in the k-d tree at a given
 * position.
 *
 * @warning This function has the side-effect of re-ordering the index array.
 *
 * @param[inout] tree The k-d tree in which to build the node.
 * @param[in] spacepoints The spacepoint array.
 * @param[inout] indices The indices from which to build an internal node.
 * @param[inout] index_buffer Temporary storage for partitioning the indices.
 * @param[in] nid The node index.
 * @param[in] pivot The requested pivot axis.
 */
__device__ bool create_internal_node(kd_tree_t tree, internal_sp_t spacepoints,
                                     const uint32_t* __restrict__ indices,
                                     uint32_t* __restrict__ index_buffer,
                                     int nid, pivot_e pivot) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();

    /*
     * Get the node and it's left and right children.
     */
    uint32_t lnid = 2 * nid + 1;
    uint32_t rnid = 2 * nid + 2;

    /*
     * Start the construction of the two children, only if we are the leader
     * thread.
     */
    if (block.thread_rank() == 0) {
        tree[lnid].type = nodetype_e::LEAF;
        tree[rnid].type = nodetype_e::LEAF;
    }

    /*
     * Let's hope this doesn't fire, but if we have zero (or god forbid a
     * negative number) nodes, we have a big problem.
     */
    assert(tree[nid].end > tree[nid].begin);

    uint32_t num_indices = tree[nid].end - tree[nid].begin;

    block.sync();

    float mid_point_value;

    /*
     * We have two different branches here; for small nodes we take an exact
     * approach, for big nodes we make approximations.
     */
    if (num_indices > block.size()) {
        /*
         * These counters are used for partitioning, and count how many points
         * we have that are greater than our pivot, as well as how many are
         * less than the pivot.
         */
        __shared__ uint32_t lower_idx, upper_idx;

        if (block.thread_rank() == 0) {
            lower_idx = 0;
            upper_idx = 0;
        }

        /*
         * Find the approximate median value of the given spacepoints in the
         * selected dimension.
         */
        float mid_point =
            get_approximate_median(spacepoints, indices, num_indices, pivot);

        block.sync();

        /*
         * Now, we partition the spacepoints (or rather, their indices) into
         * lower and upper points, counting the number of elements we have in
         * each.
         */
        for (uint32_t i = block.thread_rank(); i < num_indices;
             i += block.size()) {
            if (get_coordinate(spacepoints, indices[i], pivot) >= mid_point) {
                index_buffer[num_indices - (1u + atomicAdd(&upper_idx, 1u))] =
                    indices[i];
            } else {
                index_buffer[atomicAdd(&lower_idx, 1u)] = indices[i];
            }
        }

        block.sync();

        /*
         * Finally, set the beginning and end of our children, as well as the
         * splitting point and the current node.
         */
        if (block.thread_rank() == 0) {
            tree[lnid].begin = tree[nid].begin;
            tree[lnid].end = tree[lnid].begin + lower_idx;

            tree[rnid].begin = tree[lnid].end;
            tree[rnid].end = tree[nid].end;

            mid_point_value = mid_point;
        }

        block.sync();
    } else {
        /*
         * For small ranges, we just get the exact median. We have a helper
         * function for this which also sorts the indices as a side-effect, and
         * so we do not actually need to do any partitioning.
         */
        uint32_t mid_point = sort_and_get_median(
            spacepoints, indices, index_buffer, num_indices, pivot);

        /*
         * Correctly set the beginning and the of our children, as well as our
         * own pivot value.
         */
        if (block.thread_rank() == 0) {
            tree[lnid].begin = tree[nid].begin;
            tree[lnid].end = tree[lnid].begin + mid_point;

            tree[rnid].begin = tree[lnid].end;
            tree[rnid].end = tree[nid].end;

            mid_point_value =
                get_coordinate(spacepoints, index_buffer[mid_point], pivot);
        }

        block.sync();
    }

    /*
     * Now that the partitioning is complete, we can have our leader thread
     * put the dots on the i's and complete this internal node.
     */
    if (block.thread_rank() == 0) {
        tree[nid].type = nodetype_e::INTERNAL;
        tree[nid].dim = pivot;
        tree[nid].mid = mid_point_value;

        tree[lnid].range = tree[nid].range;
        tree[rnid].range = tree[nid].range;

        /*
         * We need to make sure that we update the ranges of our children!
         */
        if (tree[nid].dim == pivot_e::Phi) {
            tree[lnid].range.phi_max = mid_point_value;
            tree[rnid].range.phi_min = mid_point_value;
        } else if (tree[nid].dim == pivot_e::R) {
            tree[lnid].range.r_max = mid_point_value;
            tree[rnid].range.r_min = mid_point_value;
        } else {
            tree[lnid].range.z_max = mid_point_value;
            tree[rnid].range.z_min = mid_point_value;
        }
    }

    /*
     * This function also always succeeds.
     */
    return true;
}

/**
 * @brief Main k-d tree construction kernel.
 *
 * A k-d tree is constructed by repeated application of this kernel. This
 * kernel uses a one-node-per-block approach. This is not optimal for early
 * iteration counts.
 *
 * @warning This kernel assumes that the tree has already been initialized
 * through the preceding kernels.
 *
 * @param[inout] tree The tree array in which to build.
 * @param[in] num_nodes The maximum size of the k-d tree.
 * @param[in] spacepoints The spacepoints to encode in the tree.
 * @param[inout] _indices The index array to use.
 * @param[inout] _index_buffer Temporary buffer space for indices.
 * @param[in] iteration The iteration count.
 */
__global__ void __launch_bounds__(512)
    construct_kd_tree_big_step(kd_tree_t tree, uint32_t num_nodes,
                               internal_sp_t spacepoints,
                               const uint32_t* __restrict__ _indices_old,
                               uint32_t* __restrict__ _indices_new,
                               uint32_t iteration,
                               uint32_t* __restrict__ work_list_current,
                               uint32_t* __restrict__ work_list_size_current,
                               uint32_t* __restrict__ work_list_next,
                               uint32_t* __restrict__ work_list_size_next) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();

    /*
     * Calculate the current node ID. Remember that we start with 1 node
     * indexed at [0], then 2 nodes at [1, 2], then 4 nodes at [3, 4, 5, 6],
     * etc. Thus, the subarray for zero-indexed iteration i starts at 2^i - 1.
     */
    assert(blockIdx.x < *work_list_size_current);

    int nid = work_list_current[blockIdx.x];

    /*
     * Note that this is not equivalent per se to the above, and that this is
     * not an error. But it does indicate that there is no work to be done.
     */
    assert(tree[nid].type == nodetype_e::LEAF);

    bool build_leaf = (tree[nid].end - tree[nid].begin) <= 8;

    block.sync();

    /*
     * Check to see whether we could fit the range assigned to this block into
     * a leaf node. If so, we construct one. Otherwise, we construct an
     * internal node.
     */
    if (build_leaf) {
        cooperative_groups::thread_block block =
            cooperative_groups::this_thread_block();

        /*
         * Set the node to a leaf, and set the number of points it contains.
         */
        if (block.thread_rank() == 0) {
            uint32_t begin = tree[nid].begin, end = tree[nid].end;

            tree[nid].type = nodetype_e::LEAF;
            tree[nid].begin = begin;
            tree[nid].end = end;
        }

        block.sync();
    } else {
        [[maybe_unused]] bool r = false;

        pivot_e pivot;

        /*
         * If we are the root node, our pivot axis is φ; otherwise, it's what-
         * ever is next after our parent's pivot axis.
         */
        if (nid == 0) {
            pivot = pivot_e::Phi;
        } else {
            pivot = next_pivot(tree[(nid - 1) / 2].dim);
        }

        /*
         * Try to create an internal node with the given pivot.
         */
        r = create_internal_node(tree, spacepoints,
                                 &_indices_old[tree[nid].begin],
                                 &_indices_new[tree[nid].begin], nid, pivot);

        /*
         * Check whether the process succeeded. It should.
         */
        assert(r);

        if (threadIdx.x == 0 && blockIdx.y == 0) {
            uint32_t idx = atomicAdd(work_list_size_next, 2u);
            work_list_next[idx] = 2 * nid + 1;
            work_list_next[idx + 1] = 2 * nid + 2;
        }
    }
}

__global__ void __launch_bounds__(96)
    construct_kd_tree_small_step(kd_tree_t tree, uint32_t num_nodes,
                                 internal_sp_t spacepoints,
                                 const uint32_t* __restrict__ _indices_old,
                                 uint32_t* __restrict__ _indices_new,
                                 uint32_t iteration,
                                 uint32_t* __restrict__ work_list_current,
                                 uint32_t* __restrict__ work_list_size_current,
                                 uint32_t* __restrict__ work_list_next,
                                 uint32_t* __restrict__ work_list_size_next) {
    cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();

    /*
     * Calculate the current node ID. Remember that we start with 1 node
     * indexed at [0], then 2 nodes at [1, 2], then 4 nodes at [3, 4, 5, 6],
     * etc. Thus, the subarray for zero-indexed iteration i starts at 2^i - 1.
     */
    assert(blockIdx.x < *work_list_size_current);

    int nid = work_list_current[blockIdx.x];

    /*
     * Note that this is not equivalent per se to the above, and that this is
     * not an error. But it does indicate that there is no work to be done.
     */
    assert(tree[nid].type == nodetype_e::LEAF);

    bool build_leaf = (tree[nid].end - tree[nid].begin) <= 8;

    block.sync();

    /*
     * Check to see whether we could fit the range assigned to this block into
     * a leaf node. If so, we construct one. Otherwise, we construct an
     * internal node.
     */
    if (build_leaf) {
        cooperative_groups::thread_block block =
            cooperative_groups::this_thread_block();

        /*
         * Set the node to a leaf, and set the number of points it contains.
         */
        if (block.thread_rank() == 0) {
            uint32_t begin = tree[nid].begin, end = tree[nid].end;

            tree[nid].type = nodetype_e::LEAF;
            tree[nid].begin = begin;
            tree[nid].end = end;
        }

        block.sync();
    } else {
        [[maybe_unused]] bool r = false;

        pivot_e pivot;

        /*
         * If we are the root node, our pivot axis is φ; otherwise, it's what-
         * ever is next after our parent's pivot axis.
         */
        if (nid == 0) {
            pivot = pivot_e::Phi;
        } else {
            pivot = next_pivot(tree[(nid - 1) / 2].dim);
        }

        /*
         * Try to create an internal node with the given pivot.
         */
        r = create_internal_node(tree, spacepoints,
                                 &_indices_old[tree[nid].begin],
                                 &_indices_new[tree[nid].begin], nid, pivot);

        /*
         * Check whether the process succeeded. It should.
         */
        assert(r);

        if (threadIdx.x == 0 && blockIdx.y == 0) {
            uint32_t idx = atomicAdd(work_list_size_next, 2u);
            work_list_next[idx] = 2 * nid + 1;
            work_list_next[idx + 1] = 2 * nid + 2;
        }
    }
}

uint32_t round_to_power_2(uint32_t i) {
    uint32_t power = 1u;

    while (power < i) {
        power *= 2u;
    }

    return power;
}

__global__ void bake_spacepoints_kernel(uint32_t n, internal_sp_t oldsps,
                                        internal_sp_t newsps,
                                        uint32_t* __restrict__ indices) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    uint32_t i = grid.thread_rank();

    if (i < n) {
        uint32_t ni = indices[i] & ~AUXILIARY_INDEX_MASK;

        newsps[i].x = oldsps[ni].x;
        newsps[i].y = oldsps[ni].y;
        newsps[i].z = oldsps[ni].z;
        newsps[i].phi = oldsps[ni].phi;
        newsps[i].radius = oldsps[ni].radius;
        newsps[i].link = oldsps[ni].link;
    }
}

std::tuple<kd_tree_owning_t, uint32_t, internal_sp_owning_t> create_kd_tree(
    vecmem::memory_resource& mr, internal_sp_owning_t&& spacepoints,
    uint32_t n_sp) {
    /*
     * Allocate space for the indices. Since each spacepoint can produce at
     * most two indices, we allocate enough space to support this theoretical
     * upper bound.
     */
    vecmem::unique_alloc_ptr<uint32_t[]> indices =
        vecmem::make_unique_alloc<uint32_t[]>(mr, 4 * n_sp);

    /*
     * Allocate an on-device counter for the number of halo points we have,
     * which starts out at zero.
     */
    vecmem::unique_alloc_ptr<uint32_t> extra_indices =
        vecmem::make_unique_alloc<uint32_t>(mr);

    CUDA_ERROR_CHECK(cudaMemset(extra_indices.get(), 0, sizeof(uint32_t)));

    /*
     * Launch the index initialization kernel, which turns n spacepoints into
     * anywhere between n and 2n indices in the index array.
     */
    std::size_t threads_per_block1 = 256;
    initialize_index_vector<<<(n_sp / threads_per_block1) +
                                  (n_sp % threads_per_block1 == 0 ? 0 : 1),
                              threads_per_block1>>>(spacepoints, indices.get(),
                                                    n_sp, extra_indices.get());

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    /*
     * Retrieve the number of halo points from the device.
     */
    uint32_t extra_indices_h;

    CUDA_ERROR_CHECK(cudaMemcpy(&extra_indices_h, extra_indices.get(),
                                sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /*
     * The total number of indices is equal to the number of spacepoints plus
     * the number of halo points.
     */
    uint32_t num_indices = n_sp + extra_indices_h;

    /*
     * In theory, the number of nodes that is needed to support N points given
     * K points per node is 2^⌈log_2⌈N / ⌊K / 2⌋⌉ + 1⌉ - 1. In practice, we
     * build potentially _very_ unbalanced trees. Thus, we need a bit more
     * space.
     */
    uint32_t largest_layer = round_to_power_2(num_indices);
    uint32_t num_nodes = 4 * largest_layer - 1;

    /*
     * Allocate space in which the k-d tree will live.
     */
    kd_tree_owning_t tree_owner(mr, num_nodes);

    vecmem::unique_alloc_ptr<uint32_t[]> work_list_1 =
        vecmem::make_unique_alloc<uint32_t[]>(mr, num_nodes);
    vecmem::unique_alloc_ptr<uint32_t[]> work_list_2 =
        vecmem::make_unique_alloc<uint32_t[]>(mr, num_nodes);
    vecmem::unique_alloc_ptr<uint32_t> work_list_size_1 =
        vecmem::make_unique_alloc<uint32_t>(mr);
    vecmem::unique_alloc_ptr<uint32_t> work_list_size_2 =
        vecmem::make_unique_alloc<uint32_t>(mr);

    CUDA_ERROR_CHECK(cudaMemset(work_list_size_1.get(), 0, sizeof(uint32_t)));
    CUDA_ERROR_CHECK(cudaMemset(work_list_size_2.get(), 0, sizeof(uint32_t)));

    /*
     * Run the initialization kernel for the tree itself, which sets the
     * initial types of the nodes.
     */
    std::size_t threads_per_block2 = 256;
    initialize_kd_tree<<<(num_nodes / threads_per_block2) +
                             (num_nodes % threads_per_block2 == 0 ? 0 : 1),
                         threads_per_block2>>>(kd_tree_t(tree_owner), num_nodes,
                                               num_indices, work_list_1.get(),
                                               work_list_size_1.get());

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    /*
     * Allocate some temporary space for the index buffer, one half for "lower"
     * points, and one for "upper" points, defined relative to some pivot
     * point. Multiple nodes may use this buffer to do their partitions at the
     * same time.
     */
    vecmem::unique_alloc_ptr<uint32_t[]> index_buffer =
        vecmem::make_unique_alloc<uint32_t[]>(mr, num_indices);

    /*
     * Repeatedly apply the k-d tree construction kernel. The iteration width
     * is the number of nodes that is processed in parallel, and is simply 2^i
     * where i is the iteration. We can stop running kernels after we cover the
     * entire last level of the tree.
     */
    uint32_t remaining = 1, iteration = 0;

    while (remaining > 0) {
        dim3 grid_size{remaining};
        uint32_t threads_per_block;

        if (iteration <= 4) {
            threads_per_block = 512u;

            construct_kd_tree_big_step<<<grid_size, threads_per_block,
                                         threads_per_block * sizeof(float)>>>(
                kd_tree_t(tree_owner), num_nodes, spacepoints, indices.get(),
                index_buffer.get(), iteration, work_list_1.get(),
                work_list_size_1.get(), work_list_2.get(),
                work_list_size_2.get());
        } else {
            threads_per_block = 96u;

            construct_kd_tree_small_step<<<grid_size, threads_per_block,
                                           threads_per_block * sizeof(float)>>>(
                kd_tree_t(tree_owner), num_nodes, spacepoints, indices.get(),
                index_buffer.get(), iteration, work_list_1.get(),
                work_list_size_1.get(), work_list_2.get(),
                work_list_size_2.get());
        }

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        iteration++;

        CUDA_ERROR_CHECK(cudaMemcpy(&remaining, work_list_size_2.get(),
                                    sizeof(uint32_t), cudaMemcpyDeviceToHost));

        std::swap(work_list_1, work_list_2);
        std::swap(work_list_size_1, work_list_size_2);
        std::swap(indices, index_buffer);

        CUDA_ERROR_CHECK(
            cudaMemset(work_list_size_2.get(), 0, sizeof(uint32_t)));
    }

    internal_sp_owning_t new_sps(mr, num_indices);

    bake_spacepoints_kernel<<<
        num_indices / 1024u + (num_indices % 1024u == 0u ? 0u : 1u), 1024u>>>(
        num_indices, internal_sp_t(spacepoints), internal_sp_t(new_sps),
        indices.get());

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return {kd_tree_owning_t{std::move(tree_owner)}, num_nodes,
            std::move(new_sps)};
}
}  // namespace traccc::cuda
