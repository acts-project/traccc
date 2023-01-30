/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/cuda/cca/component_connection.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/allocator.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"

namespace {
static constexpr std::size_t MAX_CELLS_PER_PARTITION = 2048;
static constexpr std::size_t THREADS_PER_BLOCK = 256;
using index_t = unsigned short;
}  // namespace

namespace traccc::cuda {
namespace details {

/*
 * Convenience structure to work with flattened data arrays instead of
 * an array/vector of cells.
 */
struct cell_container {
    std::size_t size = 0;
    channel_id* channel0 = nullptr;
    channel_id* channel1 = nullptr;
    scalar* activation = nullptr;
    scalar* time = nullptr;
    geometry_id* module_id = nullptr;
};

/*
 * Convenience structure to work with flattened data arrays instead of
 * an array/vector of measures.
 */
struct measurement_container {
    unsigned int size = 0;
    scalar* channel0 = nullptr;
    scalar* channel1 = nullptr;
    scalar* variance0 = nullptr;
    scalar* variance1 = nullptr;
    geometry_id* module_id = nullptr;
};

/*
 * Check if two cells are considered close enough to be part of the same
 * cluster.
 */
__device__ bool is_adjacent(channel_id ac0, channel_id ac1, channel_id bc0,
                            channel_id bc1) {
    unsigned int p0 = (ac0 - bc0);
    unsigned int p1 = (ac1 - bc1);

    return p0 * p0 <= 1 && p1 * p1 <= 1;
}

__device__ void reduce_problem_cell(cell_container& cells, index_t tid,
                                    unsigned char& adjc, index_t adjv[]) {
    /*
     * The number of adjacent cells for each cell must start at zero, to
     * avoid uninitialized memory. adjv does not need to be zeroed, as
     * we will only access those values if adjc indicates that the value
     * is set.
     */
    adjc = 0;

    channel_id c0 = cells.channel0[tid];
    channel_id c1 = cells.channel1[tid];
    geometry_id gid = cells.module_id[tid];

    /*
     * First, we traverse the cells backwards, starting from the current
     * cell and working back to the first, collecting adjacent cells
     * along the way.
     */
    for (index_t j = tid - 1; j < tid; --j) {
        /*
         * Since the data is sorted, we can assume that if we see a cell
         * sufficiently far away in both directions, it becomes
         * impossible for that cell to ever be adjacent to this one.
         * This is a small optimisation.
         */
        if (cells.channel1[j] + 1 < c1 || cells.module_id[j] != gid) {
            break;
        }

        /*
         * If the cell examined is adjacent to the current cell, save it
         * in the current cell's adjacency set.
         */
        if (is_adjacent(c0, c1, cells.channel0[j], cells.channel1[j])) {
            adjv[adjc++] = j;
        }
    }

    /*
     * Now we examine all the cells past the current one, using almost
     * the same logic as in the backwards pass.
     */
    for (index_t j = tid + 1; j < cells.size; ++j) {
        /*
         * Note that this check now looks in the opposite direction! An
         * important difference.
         */
        if (cells.channel1[j] > c1 + 1 || cells.module_id[j] != gid) {
            break;
        }

        if (is_adjacent(c0, c1, cells.channel0[j], cells.channel1[j])) {
            adjv[adjc++] = j;
        }
    }
}

/*
 * Implementation of a FastSV algorithm with the following steps:
 *   1) mix of stochastic and aggressive hooking
 *   2) shortcutting
 *
 * The implementation corresponds to an adapted versiion of Algorithm 3 of
 * the following paper:
 * https://www.sciencedirect.com/science/article/pii/S0743731520302689
 *
 * f      = array holding the parent cell ID for the current iteration.
 * gf     = array holding grandparent cell ID from the previous iteration.
            This array only gets updated at the end of the iteration to prevent
            race conditions.
 */
__device__ void fast_sv_1(index_t* f, index_t* gf, unsigned char adjc[],
                          index_t adjv[][8], unsigned int size) {
    /*
     * The algorithm finishes if an iteration leaves the arrays unchanged.
     * This varible will be set if a change is made, and dictates if another
     * loop is necessary.
     */
    bool gf_changed;

    do {
        /*
         * Reset the end-parameter to false, so we can set it to true if we
         * make a change to the gf array.
         */
        gf_changed = false;

        /*
         * The algorithm executes in a loop of three distinct parallel
         * stages. In this first one, a mix of stochastic and aggressive
         * hooking, we examine adjacent cells and copy their grand parents
         * cluster ID if it is lower than ours, essentially merging the two
         * together.
         */
        for (index_t tst = 0, tid;
             (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
            __builtin_assume(adjc[tst] <= 8);

            for (unsigned char k = 0; k < adjc[tst]; ++k) {
                index_t q = gf[adjv[tst][k]];

                if (gf[tid] > q) {
                    f[f[tid]] = q;
                    f[tid] = q;
                }
            }
        }

        /*
         * Each stage in this algorithm must be preceded by a
         * synchronization barrier!
         */
        __syncthreads();

        /*
         * The second stage is shortcutting, which is an optimisation that
         * allows us to look at any shortcuts in the cluster IDs that we
         * can merge without adjacency information.
         */
        for (index_t tid = threadIdx.x; tid < size; tid += blockDim.x) {
            if (f[tid] > gf[tid]) {
                f[tid] = gf[tid];
            }
        }

        /*
         * Synchronize before the final stage.
         */
        __syncthreads();

        /*
         * Update the array for the next generation, keeping track of any
         * changes we make.
         */
        for (index_t tid = threadIdx.x; tid < size; tid += blockDim.x) {
            if (gf[tid] != f[f[tid]]) {
                gf[tid] = f[f[tid]];
                gf_changed = true;
            }
        }

        /*
         * To determine whether we need another iteration, we use block
         * voting mechanics. Each thread checks if it has made any changes
         * to the arrays, and votes. If any thread votes true, all threads
         * will return a true value and go to the next iteration. Only if
         * all threads return false will the loop exit.
         */
    } while (__syncthreads_or(gf_changed));
}

/*
 * Implementation of a FastSV algorithm with the following steps:
 *   1) stochastic hooking
 *   2) aggressive hooking
 *   3) shortcutting
 *
 * The implementation corresponds to Algorithm 2 of the following paper:
 * https://epubs.siam.org/doi/pdf/10.1137/1.9781611976137.5
 *
 * f      = array holding the parent cell ID for the current iteration.
 * f_next = buffer array holding updated information for the next iteration.
 */
__device__ void fast_sv_2(index_t* f, index_t* f_next, unsigned char adjc[],
                          index_t adjv[][8], unsigned int size) {
    /*
     * The algorithm finishes if an iteration leaves the array for the next
     * iteration unchanged.
     * This varible will be set if a change is made, and dictates if another
     * loop is necessary.
     */
    bool f_next_changed;

    do {
        /*
         * Reset the end-parameter to false, so we can set it to true if we
         * make a change to the f_next array.
         */
        f_next_changed = false;

        /*
         * The algorithm executes in a loop of four distinct parallel
         * stages. In this first one, stochastic hooking, we examine the
         * grandparents of adjacent cells and copy cluster ID if it
         * is lower than our, essentially merging the two together.
         */
        for (index_t tst = 0, tid;
             (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
            for (unsigned char k = 0; k < adjc[tst]; ++k) {
                index_t q = f[f[adjv[tst][k]]];

                if (q < f_next[f[tid]]) {
                    // hook to grandparent of adjacent cell
                    f_next[f[tid]] = q;
                    f_next_changed = true;
                }
            }
        }

        /*
         * Synchronize before the next stage.
         */
        __syncthreads();

        /*
         * The second stage performs aggressive hooking, during which each
         * cell might be hooked to the grand parent of an adjacent cell.
         */
        for (index_t tst = 0, tid;
             (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
            for (unsigned char k = 0; k < adjc[tst]; ++k) {
                index_t q = f[f[adjv[tst][k]]];

                if (q < f_next[tid]) {
                    f_next[tid] = q;
                    f_next_changed = true;
                }
            }
        }

        /*
         * Synchronize before the next stage.
         */
        __syncthreads();

        /*
         * The third stage is shortcutting, which is an optimisation that
         * allows us to look at any shortcuts in the cluster IDs that we
         * can merge without adjacency information.
         */
        for (index_t tst = 0, tid;
             (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
            if (f[f[tid]] < f_next[tid]) {
                f_next[tid] = f[f[tid]];
                f_next_changed = true;
            }
        }

        /*
         * Synchronize before the final stage.
         */
        __syncthreads();

        /*
         * Update the array for the next generation.
         */
        for (index_t tst = 0, tid;
             (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
            f[tid] = f_next[tid];
        }

        /*
         * To determine whether we need another iteration, we use block
         * voting mechanics. Each thread checks if it has made any changes
         * to the arrays, and votes. If any thread votes true, all threads
         * will return a true value and go to the next iteration. Only if
         * all threads return false will the loop exit.
         */
    } while (__syncthreads_or(f_next_changed));
}

/*
 * Implementation of a simplified SV algorithm with the following steps:
 *   1) tree hooking
 *   2) shortcutting
 *
 * The implementation corresponds to Algorithm 1 of the following paper:
 * https://epubs.siam.org/doi/pdf/10.1137/1.9781611976137.5
 *
 * f      = array holding the parent cell ID for the current iteration.
 * f_next = buffer array holding updated information for the next iteration.
 */
__device__ void simplified_sv(index_t* f, index_t* f_next, unsigned char adjc[],
                              index_t adjv[][8], unsigned int size) {
    /*
     * The algorithm finishes if an iteration leaves the array for the next
     * iteration unchanged.
     * This varible will be set if a change is made, and dictates if another
     * loop is necessary.
     */
    bool f_changed;

    do {
        /*
         * Reset the end-parameter to false, so we can set it to true if we
         * make a change to the f_next array.
         */
        f_changed = false;

        /*
         * The algorithm executes in a loop of four distinct parallel
         * stages. In this first one, tree hooking, we examine adjacent cells of
         * cluster roots and copy their cluster ID if it is lower than our,
         * essentially merging the two together.
         */
        for (index_t tst = 0, tid;
             (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
            if (f[tid] == f[f[tid]]) {  // only perform for roots of clusters
                for (unsigned char k = 0; k < adjc[tst]; ++k) {
                    index_t q = f[adjv[tst][k]];
                    if (q < f[tid]) {
                        f_next[f[tid]] = q;
                        f_changed = true;
                    }
                }
            }
        }

        /*
         * Synchronize before the next stage.
         */
        __syncthreads();

        /*
         * Update the array for the next stage of the iteration.
         */
        for (index_t tst = 0, tid;
             (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
            f[tid] = f_next[tid];
        }

        /*
         * Synchronize before the next stage.
         */
        __syncthreads();

        /*
         * The third stage is shortcutting, which is an optimisation that
         * allows us to look at any shortcuts in the cluster IDs that we
         * can merge without adjacency information.
         */
        for (index_t tst = 0, tid;
             (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
            if (f[f[tid]] < f[tid]) {
                f_next[tid] = f[f[tid]];
                f_changed = true;
            }
        }

        /*
         * Synchronize before the final stage.
         */
        __syncthreads();

        /*
         * Update the array for the next generation.
         */
        for (index_t tst = 0, tid;
             (tid = tst * blockDim.x + threadIdx.x) < size; ++tst) {
            f[tid] = f_next[tid];
        }

        /*
         * To determine whether we need another iteration, we use block
         * voting mechanics. Each thread checks if it has made any changes
         * to the arrays, and votes. If any thread votes true, all threads
         * will return a true value and go to the next iteration. Only if
         * all threads return false will the loop exit.
         */
    } while (__syncthreads_or(f_changed));
}

__device__ void aggregate_clusters(const cell_container& cells,
                                   measurement_container& out, index_t* f) {
    __shared__ unsigned int outi;

    if (threadIdx.x == 0) {
        outi = 0;
    }

    __syncthreads();

    /*
     * This is the post-processing stage, where we merge the clusters into a
     * single measurement and write it to the output.
     */
    for (index_t tst = 0, tid;
         (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {

        /*
         * If and only if the value in the work arrays is equal to the index
         * of a cell, that cell is the "parent" of a cluster of cells. If
         * they are not, there is nothing for us to do. Easy!
         */
        if (f[tid] == tid) {
            /*
             * If we are a cluster owner, atomically claim a position in the
             * output array which we can write to.
             */
            unsigned int id = atomicAdd(&outi, 1);

            /*
             * These variables keep track of the sums of X and Y coordinates
             * for the final coordinates, the total activation weight, as
             * well as the sum of squares of positions, which we use to
             * calculate the variance.
             */
            float sw = 0.0;
            float mx = 0.0, my = 0.0;
            float vx = 0.0, vy = 0.0;

            /*
             * Now, we iterate over all other cells to check if they belong
             * to our cluster. Note that we can start at the current index
             * because no cell is every a child of a cluster owned by a cell
             * with a higher ID.
             */
            for (index_t j = tid; j < cells.size; j++) {
                /*
                 * If the value of this cell is equal to our, that means it
                 * is part of our cluster. In that case, we take its values
                 * for position and add them to our accumulators.
                 */
                if (f[j] == tid) {
                    float w = cells.activation[j];

                    sw += w;

                    float pmx = mx, pmy = my;
                    float dx = cells.channel0[j] - pmx;
                    float dy = cells.channel1[j] - pmy;
                    float wf = w / sw;

                    mx = pmx + wf * dx;
                    my = pmy + wf * dy;

                    vx += w * dx * (cells.channel0[j] - mx);
                    vy += w * dy * (cells.channel1[j] - my);
                }
            }

            /*
             * Write the average weighted x and y coordinates, as well as
             * the weighted average square position, to the output array.
             */
            out.channel0[id] = mx;
            out.channel1[id] = my;
            out.variance0[id] = vx / sw;
            out.variance1[id] = vy / sw;
            out.module_id[id] = cells.module_id[tid];
        }
    }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK) void ccl_kernel(
    const cell_container container, const unsigned* partitions,
    measurement_container& _out_ctnr) {
    const unsigned start = partitions[blockIdx.x];

    /*
     * Seek the correct cell region in the input data. Again, this is all a
     * contiguous block of memory for now, and we use the blocks array to
     * define the different ranges per block/module. At the end of this we
     * have the starting address of the block of cells dedicated to this
     * module, and we have its size.
     */
    cell_container cells;
    cells.size = partitions[blockIdx.x + 1] - partitions[blockIdx.x];
    cells.channel0 = &container.channel0[start];
    cells.channel1 = &container.channel1[start];
    cells.activation = &container.activation[start];
    cells.time = &container.time[start];
    cells.module_id = &container.module_id[start];

    assert(cells.size <= MAX_CELLS_PER_PARTITION);

    /*
     * As an optimisation, we will keep track of which cells are adjacent to
     * each other cell. To do this, we define, in thread-local memory or
     * registers, up to eight adjacent cell indices and we keep track of how
     * many adjacent cells there are (i.e. adjc[i] determines how many of
     * the eight values in adjv[i] are actually meaningful).
     *
     * The implementation is such that a thread might need to process more
     * than one hit. As such, we keep one counter and eight indices _per_
     * hit the thread is processing. This number is never larger than
     * the max number of activations per module divided by the threads per
     * block.
     *
     * adjc = adjecency count
     * adjv = adjecency vector
     */
    index_t adjv[MAX_CELLS_PER_PARTITION / THREADS_PER_BLOCK][8];
    unsigned char adjc[MAX_CELLS_PER_PARTITION / THREADS_PER_BLOCK];

    /*
     * After this is all done, we synchronise the block. I am not absolutely
     * certain that this is necessary here, but the overhead is not that big
     * and we might as well be safe rather than sorry.
     */
    __syncthreads();

    /*
     * This loop initializes the adjacency cache, which essentially
     * translates the sparse CCL problem into a graph CCL problem which we
     * can tackle with well-studied algorithms. This loop pattern is often
     * found throughout this code. We iterate over the number of activations
     * each thread must process. Sadly, the CUDA limit is 1024 threads per
     * block and we cannot guarantee that there will be fewer than 1024
     * activations in a module. So each thread must be able to do more than
     * one.
     */
    for (index_t tst = 0, tid;
         (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
        reduce_problem_cell(cells, tid, adjc[tst], adjv[tst]);
    }

    // if (threadIdx.x == 0) asm("mov.u32 %0, %clock;" : "=r"(c2) );

    /*
     * These arrays are the meat of the pudding of this algorithm, and we
     * will constantly be writing and reading from them which is why we
     * declare them to be in the fast shared memory. Note that this places a
     * limit on the maximum activations per module, as the amount of shared
     * memory is limited. These could always be moved to global memory, but
     * the algorithm would be decidedly slower in that case.
     */
    __shared__ index_t f[MAX_CELLS_PER_PARTITION],
        f_next[MAX_CELLS_PER_PARTITION];

    for (index_t tst = 0, tid;
         (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
        /*
         * At the start, the values of f and f_next should be equal to the
         * ID of the cell.
         */
        f[tid] = tid;
        f_next[tid] = tid;
    }

    /*
     * Now that the data has initialized, we synchronize again before we
     * move onto the actual processing part.
     */
    __syncthreads();

    fast_sv_1(f, f_next, adjc, adjv, cells.size);

    /*
     * This variable will be used to write to the output later.
     */
    __shared__ unsigned int outi;

    /*
     * Initialize the counter of clusters per thread block
     */
    if (threadIdx.x == 0) {
        outi = 0;
    }

    __syncthreads();

    /*
     * Count the number of clusters by checking how many cells have
     * themself assigned as a parent.
     */
    for (index_t tst = 0, tid;
         (tid = tst * blockDim.x + threadIdx.x) < cells.size; ++tst) {
        if (f[tid] == tid) {
            atomicAdd(&outi, 1);
        }
    }

    __syncthreads();

    /*
     * Add the number of clusters of each thread block to the total
     * number of clusters. At the same time, a cluster id is retrieved
     * for the next data processing step.
     * Note that this might be not the same cluster as has been treated
     * previously. However, since each thread block spawns a the maximum
     * amount of threads per block, this has no sever implications.
     */
    if (threadIdx.x == 0) {
        outi = atomicAdd(&_out_ctnr.size, outi);
    }

    __syncthreads();

    measurement_container out;
    out.channel0 = &_out_ctnr.channel0[outi];
    out.channel1 = &_out_ctnr.channel1[outi];
    out.variance0 = &_out_ctnr.variance0[outi];
    out.variance1 = &_out_ctnr.variance1[outi];
    out.module_id = &_out_ctnr.module_id[outi];

    aggregate_clusters(cells, out, f);
}

std::tuple<vecmem::unique_alloc_ptr<unsigned[]>, std::size_t> partition_cpu(
    const cell_container_types::host& data, vecmem::memory_resource& mem,
    const details::cell_container cells) {
    vecmem::unique_alloc_ptr<unsigned[]> partitions =
        vecmem::make_unique_alloc<unsigned[]>(mem, cells.size);
    std::size_t index = 0;
    std::size_t size = 0;
    std::size_t elements = 0;
    std::size_t pidx = 0;

    /*
     * Iterate over every cell module in the current data set.
     */
    for (std::size_t i = 0; i < data.size(); ++i) {
        /*
         * We start at 0 since this is the origin of the local coordinate
         * system within a cell module.
         */
        channel_id last_mid = 0;

        for (const cell& c : data.at(i).items) {
            /*
             * Create a new partition if an "empty" row is detected. A row
             * is considered "empty" if the channel1 value between two
             * consecutive cells have a difference > 1.
             * To prevent creating many small partitions, the current partition
             * must have at least twice the size of threads per block. This
             * guarantees that each thread handles later at least two cells.
             */
            if (c.channel1 > last_mid + 1 && size >= 2 * THREADS_PER_BLOCK) {
                partitions[pidx++] = index;

                index += size;
                size = 0;
            }

            last_mid = c.channel1;
            size += 1;
            elements += 1;
        }

        /*
         * If a cell module has many activations and therefore no empty
         * rows, it is possible that partitions reach a considerable
         * size. To prevent very big partitions, we check at the end of each
         * module if the current partition is not above a threshold, and end the
         * current partition if necessary here.
         */
        if (size >= 2 * THREADS_PER_BLOCK) {
            partitions[pidx++] = index;

            index += size;
            size = 0;
        }
    }

    /*
     * Create the very last partition after having iterated over all cell
     * modules and cells.
     */
    if (size > 0) {
        partitions[pidx++] = index;
    }

    partitions[pidx++] = elements;

    return {std::move(partitions), pidx};
}

__global__ void partition_kernel(const cell_container cells, unsigned* out,
                                 unsigned long long int* idx, unsigned slots) {
    /*
     * We will use shared memory as intermediate storage for our partitions.
     * All of this is mostly setup.
     */
    extern __shared__ unsigned tmp[];
    __shared__ unsigned tmp_idx;
    __shared__ unsigned out_idx;

    if (threadIdx.x == 0) {
        tmp_idx = 0;
    }

    __syncthreads();

    /*
     * In the first segment of the kernel, we will identify all cells for which
     * the next cell skips a row, or is on a different module. This marks a
     * valid partition point, even if this leads to an extremely fine
     * partition.
     */
    for (unsigned cid = blockIdx.x * slots + threadIdx.x;
         cid < (blockIdx.x + 1) * slots; cid += blockDim.x) {
        if (cid == 0 || cid == cells.size) {
            /*
             * We always need a partition that starts at the beginning, and a
             * trailing partition at the end. This clause ensures that.
             */
            tmp[atomicAdd(&tmp_idx, 1u)] = cid;
        } else if (cid + 1 < cells.size &&
                   (cells.channel1[cid + 1] > cells.channel1[cid] + 1 ||
                    cells.module_id[cid + 1] != cells.module_id[cid])) {
            /*
             * In this case, we have found an intermediate partition point: a
             * switch to a new module, or the next hit is more than a full row
             * away!
             */
            tmp[atomicAdd(&tmp_idx, 1u)] = cid + 1;
        }
    }

    __syncthreads();

    /*
     * We proceed with the next segment. The first segment finds partition
     * points, but the GPU does not guarantee that warps execute in order, so
     * the partitions may be scrambled. This implementation of odd-even sort
     * quickly sorts them.
     */
    bool sorted;

    do {
        sorted = true;

        /*
         * Odd component.
         */
        for (uint32_t j = 2 * threadIdx.x + 1; j + 1 < tmp_idx;
             j += 2 * blockDim.x) {

            if (tmp[j] > tmp[j + 1]) {
                unsigned k = tmp[j];
                tmp[j] = tmp[j + 1];
                tmp[j + 1] = k;
                sorted = false;
            }
        }

        __syncthreads();

        /*
         * Even component.
         */
        for (uint32_t j = 2 * threadIdx.x; j + 1 < tmp_idx;
             j += 2 * blockDim.x) {
            if (tmp[j] > tmp[j + 1]) {
                unsigned k = tmp[j];
                tmp[j] = tmp[j + 1];
                tmp[j + 1] = k;
                sorted = false;
            }
        }

        /*
         * We keep running until no thread reports that the array is unsorted!
         */
    } while (__syncthreads_or(!sorted));

    /*
     * Next, we will combine partitions to more evenly spread the load on the
     * actual CCL kernel.
     *
     * This code works by overriding the existing array of partition indices.
     * The `old_idx` variable denotes the end of the old array, the `base_idx`
     * variable denotes the current starting index in the old array, and the
     * `tmp_idx` variable denotes the index we write partitions to in the new
     * array. The old and new array are actually the same memory, but the
     * writing index for the new points will always be behind the reading
     * indices in the old part, so this is safe!
     */
    const unsigned old_idx = tmp_idx;

    __syncthreads();

    /*
     * Note that the first element always remains as it is, so we can simply
     * start the process from index 1; that means the first element is never
     * touched.
     */
    unsigned base_idx = 1;

    if (threadIdx.x == 0) {
        tmp_idx = 1;
    }

    __syncthreads();

    /*
     * Now, try to merge partitions. Note once again that the base index is the
     * position we look at in the old array. If this reaches the old index, we
     * have reached the final point in the array and we are done.
     */
    while (base_idx < old_idx) {
        /*
         * Retrieve the cell index of the last partition in the new segment
         * of the array; we will compare against this point to check whether
         * the size of the partition conforms with the maximum size.
         */
        unsigned base_val = tmp[tmp_idx - 1];

        /*
         * Each thread might need to check multiple partitions. We check in
         * blocks starting from the beginning of the array and moving towards
         * the end of it.
         */
        unsigned j = 0;
        int rem;

        do {
            unsigned i = base_idx + j * blockDim.x + threadIdx.x;

            /*
             * Each thread computes whether the partition it is investigating
             * lies within the boundaries of the permissible partition size.
             * A useful consequence of this is that the number of threads that
             * satisfy this condition is also the delta that we must apply to
             * the index to find the first partition that does _not_ satify the
             * requirement. This works because the array is sorted.
             *
             * In case all threads report that they are within reach, the split
             * may be in the next chunk. Thus, we consider a return equal to
             * the size of the block to mean that we need to try this process
             * again.
             */
            rem = __syncthreads_count(
                i + 1 < old_idx && tmp[i] < base_val + MAX_CELLS_PER_PARTITION);

            ++j;
        } while (rem == blockDim.x);

        /*
         * Compute the new base index.
         */
        base_idx += (j - 1) * blockDim.x + rem + 1;

        /*
         * The lead thread inserts the partition into the new array.
         */
        if (threadIdx.x == 0) {
            tmp[tmp_idx++] = tmp[base_idx - 1];
        }

        __syncthreads();
    }

    /*
     * Next, we reserve space in the output array in global memory.
     */
    if (threadIdx.x == 0) {
        if (tmp_idx > 0) {
            out_idx = atomicAdd(idx, tmp_idx);
        }
    }

    __syncthreads();

    /*
     * The remaining threads now wake up, and all threads proceed to write the
     * array of partitions from shared memory to global memory in a coalesced
     * fashion.
     */
    for (unsigned i = threadIdx.x; i < tmp_idx; i += blockDim.x) {
        out[out_idx + i] = tmp[i];
    }
}

__global__ void partition_sorting_kernel(unsigned* out,
                                         const unsigned long long int* count) {
    /*
     * This should only EVER be launched with a single block!
     */
    assert(gridDim.x == 1);

    /*
     * Another implementation of odd-even sorting. But can I say, despite its
     * O(n^2) worst case performance it's perfect for sorting small arrays on
     * parallel shared memory machines!
     */
    bool sorted;

    do {
        sorted = true;

        for (uint32_t j = 2 * threadIdx.x + 1; j + 1 < *count;
             j += 2 * blockDim.x) {

            if (out[j] > out[j + 1]) {
                unsigned k = out[j];
                out[j] = out[j + 1];
                out[j + 1] = k;
                sorted = false;
            }
        }

        __syncthreads();

        for (uint32_t j = 2 * threadIdx.x; j + 1 < *count;
             j += 2 * blockDim.x) {
            if (out[j] > out[j + 1]) {
                unsigned k = out[j];
                out[j] = out[j + 1];
                out[j + 1] = k;
                sorted = false;
            }
        }
    } while (__syncthreads_or(!sorted));
}

std::tuple<vecmem::unique_alloc_ptr<unsigned[]>, std::size_t> partition_gpu(
    const cell_container_types::host& data, vecmem::memory_resource& mem,
    const details::cell_container cells) {
    /*
     * First, we allocate memory for our partitions, as well as memory for
     * an integer in which to store the partition counts.
     */
    vecmem::unique_alloc_ptr<unsigned[]> partitions =
        vecmem::make_unique_alloc<unsigned[]>(mem, cells.size + 1);
    vecmem::unique_alloc_ptr<unsigned long long int> pidx =
        vecmem::make_unique_alloc<unsigned long long int>(mem);

    /*
     * The partition counter must be set to zero.
     */
    CUDA_ERROR_CHECK(cudaMemset(pidx.get(), 0, sizeof(unsigned long long int)));

    /*
     * The partitioning kernel merges partitions within the same thread block.
     * This works better, in principle, if there are more partitions to
     * examine, because it reduces fragmentation of partitions. This means that
     * it is sometimes desirable to process more than one cell per thread. This
     * slots variable determines the number of cells that is examined per
     * block.
     */
    const unsigned slots = 512;

    /*
     * Launch the actual partitioning kernel and wait for it to finish.
     */
    const int grid_size =
        std::max(1ul, cells.size / slots + (cells.size % slots == 0 ? 0 : 1));
    const int blck_size = 256;
    const int smem_size = slots * sizeof(unsigned);

    partition_kernel<<<grid_size, blck_size, smem_size>>>(
        cells, partitions.get(), pidx.get(), slots);

    CUDA_ERROR_CHECK(cudaPeekAtLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    /*
     * Next, we need to make sure that the partitions are sorted. Because there
     * are usually very few partitions (less than 1000) we can do this fairly
     * efficiently with a single block running odd-even sort.
     */
    partition_sorting_kernel<<<1, 1024>>>(partitions.get(), pidx.get());

    CUDA_ERROR_CHECK(cudaPeekAtLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    /*
     * Finally, we copy the number of partitions back to the host.
     *
     * TODO: Replace this with dynamic parallelism to obviate the need for the
     * copy back to the host.
     */
    unsigned long long int hpidx;

    CUDA_ERROR_CHECK(cudaMemcpy(&hpidx, pidx.get(),
                                sizeof(unsigned long long int),
                                cudaMemcpyDeviceToHost));

    return {std::move(partitions), hpidx};
}
}  // namespace details

component_connection::output_type component_connection::operator()(
    const cell_container_types::host& data) const {
    vecmem::cuda::managed_memory_resource upstream;
    vecmem::cuda::device_memory_resource dmem;
    vecmem::binary_page_memory_resource mem(upstream);

    std::size_t total_cells = 0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        total_cells += data.at(i).items.size();
    }

    /*
     * Flatten the data to handle memory access (fetch and cache)
     * more efficiently. This removes the hierarchy level that
     * references to the cell module.
     */
    vecmem::vector<channel_id> channel0(&mem);
    channel0.reserve(total_cells);
    vecmem::vector<channel_id> channel1(&mem);
    channel1.reserve(total_cells);
    vecmem::vector<scalar> activation(&mem);
    activation.reserve(total_cells);
    vecmem::vector<scalar> time(&mem);
    time.reserve(total_cells);
    vecmem::vector<geometry_id> module_id(&mem);
    module_id.reserve(total_cells);

    for (std::size_t i = 0; i < data.size(); ++i) {
        for (std::size_t j = 0; j < data.at(i).items.size(); ++j) {
            channel0.push_back(data.at(i).items.at(j).channel0);
            channel1.push_back(data.at(i).items.at(j).channel1);
            activation.push_back(data.at(i).items.at(j).activation);
            time.push_back(data.at(i).items.at(j).time);
            module_id.push_back(data.at(i).header.module);
        }
    }

    /*
     * Store the flattened arrays in a convenience data container.
     */
    details::cell_container container;
    container.size = total_cells;
    container.channel0 = channel0.data();
    container.channel1 = channel1.data();
    container.activation = activation.data();
    container.time = time.data();
    container.module_id = module_id.data();

    /*
     * Separate the problem into various subproblems (partitions).
     * We know that the input data is sorted primarily on channel1 (y-axis),
     * and secondarily on channel0 (x-axis). This allows the cheap creation
     * of partitions based on the distance of the y-value between two
     * consecutive cells. If this distance is above a threshold, we have the
     * guarantee that the two cells belong not to the same cluster.
     *
     * Runs on the GPU, but a CPU implementation is also available!
     */
    std::tuple<vecmem::unique_alloc_ptr<unsigned[]>, std::size_t> partitions =
        details::partition_gpu(data, dmem, container);

    /*
     * Reserve space for the result of the algorithm. Currently, there is
     * enough space allocated that (in theory) each cell could be a single
     * cluster, but this should not be the case with real experiment data.
     */
    vecmem::allocator alloc(mem);

    details::measurement_container* mctnr =
        alloc.new_object<details::measurement_container>();

    mctnr->channel0 = static_cast<scalar*>(
        alloc.allocate_bytes(total_cells * sizeof(scalar)));
    mctnr->channel1 = static_cast<scalar*>(
        alloc.allocate_bytes(total_cells * sizeof(scalar)));
    mctnr->variance0 = static_cast<scalar*>(
        alloc.allocate_bytes(total_cells * sizeof(scalar)));
    mctnr->variance1 = static_cast<scalar*>(
        alloc.allocate_bytes(total_cells * sizeof(scalar)));
    mctnr->module_id = static_cast<geometry_id*>(
        alloc.allocate_bytes(total_cells * sizeof(geometry_id)));

    /*
     * Run the connected component labeling algorithm to retrieve the clusters.
     *
     * This step includes the measurement (hit) creation for each cluster.
     */
    if (std::get<1>(partitions) > 1) {
        ccl_kernel<<<std::get<1>(partitions) - 1, THREADS_PER_BLOCK>>>(
            container, std::get<0>(partitions).get(), *mctnr);

        CUDA_ERROR_CHECK(cudaPeekAtLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }

    /*
     * Copy back the data from our flattened data structure into the traccc EDM.
     */
    output_type out;

    for (std::size_t i = 0; i < data.size(); ++i) {
        vecmem::vector<measurement> v(&mem);

        for (std::size_t j = 0; j < mctnr->size; ++j) {
            if (mctnr->module_id[j] == data.at(i).header.module) {
                measurement m;

                m.local = {mctnr->channel0[j], mctnr->channel1[j]};
                m.variance = {mctnr->variance0[j], mctnr->variance1[j]};

                v.push_back(m);
            }
        }

        out.push_back(cell_module(data.at(i).header), std::move(v));
    }

    return out;
}
}  // namespace traccc::cuda
