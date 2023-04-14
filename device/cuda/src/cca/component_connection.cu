/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/cuda/cca/component_connection.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/allocator.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"

namespace {
static constexpr std::size_t MAX_CELLS_PER_PARTITION = 2048;
static constexpr std::size_t TARGET_CELLS_PER_PARTITION = 1024;
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
    unsigned int* module_link = nullptr;
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
    unsigned int* module_link = nullptr;
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
    unsigned int mlink = cells.module_link[tid];

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
        if (cells.channel1[j] + 1 < c1 || cells.module_link[j] != mlink) {
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
        if (cells.channel1[j] > c1 + 1 || cells.module_link[j] != mlink) {
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
            out.module_link[id] = cells.module_link[tid];
        }
    }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK) void ccl_kernel(
    const cell_container container, measurement_container& _out_ctnr,
    unsigned long num_cells) {
    __shared__ unsigned start, end;

    /*
     * First, we determine the exact range of cells that is to be examined by
     * this block of threads. We start from an initial range determined by the
     * block index multiplied by the target number of cells per block. We then
     * shift both the start and the end of the block forward (to a later point
     * in the array); start and end may be moved different amounts.
     */
    if (threadIdx.x == 0) {
        /*
         * Start off by naively determining the size of this block's partition.
         */
        start = blockIdx.x * TARGET_CELLS_PER_PARTITION;
        end =
            std::min(num_cells, (blockIdx.x + 1) * TARGET_CELLS_PER_PARTITION);

        /*
         * Next, shift the starting point to a position further in the array;
         * the purpose of this is to ensure that we are not operating on any
         * cells that have been claimed by the previous block (if any).
         */
        while (start != 0 &&
               container.module_link[start - 1] ==
                   container.module_link[start] &&
               container.channel1[start] <= container.channel1[start - 1] + 1) {
            ++start;
        }

        /*
         * Then, claim as many cells as we need past the naive end of the
         * current block to ensure that we do not end our partition on a cell
         * that is not a possible boundary!
         */
        while (end < num_cells &&
               container.module_link[end - 1] == container.module_link[end] &&
               container.channel1[end] <= container.channel1[end - 1] + 1) {
            ++end;
        }
    }

    __syncthreads();

    /*
     * Seek the correct cell region in the input data. Again, this is all a
     * contiguous block of memory for now, and we use the blocks array to
     * define the different ranges per block/module. At the end of this we
     * have the starting address of the block of cells dedicated to this
     * module, and we have its size.
     */
    cell_container cells;
    cells.size = end - start;
    cells.channel0 = &container.channel0[start];
    cells.channel1 = &container.channel1[start];
    cells.activation = &container.activation[start];
    cells.time = &container.time[start];
    cells.module_link = &container.module_link[start];

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
    out.module_link = &_out_ctnr.module_link[outi];

    aggregate_clusters(cells, out, f);
}
}  // namespace details

component_connection::output_type component_connection::operator()(
    const cell_collection_types::host& cells) const {
    vecmem::cuda::managed_memory_resource upstream;
    vecmem::cuda::device_memory_resource dmem;
    vecmem::binary_page_memory_resource mem(upstream);

    std::size_t total_cells = cells.size();

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
    vecmem::vector<unsigned int> module_link(&mem);
    module_link.reserve(total_cells);

    for (std::size_t i = 0; i < cells.size(); ++i) {
        channel0.push_back(cells.at(i).channel0);
        channel1.push_back(cells.at(i).channel1);
        activation.push_back(cells.at(i).activation);
        time.push_back(cells.at(i).time);
        module_link.push_back(cells.at(i).module_link);
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
    container.module_link = module_link.data();

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
    mctnr->module_link = static_cast<unsigned int*>(
        alloc.allocate_bytes(total_cells * sizeof(unsigned int)));

    /*
     * Run the connected component labeling algorithm to retrieve the clusters.
     *
     * This step includes the measurement (hit) creation for each cluster.
     */
    ccl_kernel<<<
        std::max(1ul,
                 (total_cells / TARGET_CELLS_PER_PARTITION) +
                     (total_cells % TARGET_CELLS_PER_PARTITION != 0 ? 1 : 0)),
        THREADS_PER_BLOCK>>>(container, *mctnr, total_cells);

    CUDA_ERROR_CHECK(cudaPeekAtLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    /*
     * Copy back the data from our SoA data structure into the traccc EDM.
     */
    output_type out;

    for (std::size_t i = 0; i < mctnr->size; ++i) {
        alt_measurement m;
        m.local = {mctnr->channel0[i], mctnr->channel1[i]};
        m.variance = {mctnr->variance0[i], mctnr->variance1[i]};
        m.module_link = mctnr->module_link[i];

        out.push_back(m);
    }

    return out;
}
}  // namespace traccc::cuda
