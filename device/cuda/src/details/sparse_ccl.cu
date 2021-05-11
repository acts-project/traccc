#include "edm/cell.hpp"
#include "edm/measurement.hpp"

#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/containers/vector.hpp"

#include "sparse_ccl.cuh"

#include <iostream>

namespace traccc { namespace cuda { namespace details {
    __device__
    bool is_adjacent(const cell & a, const cell & b) {
        return (
            (a.channel0 - b.channel0)*(a.channel0 - b.channel0) <= 1 &&
            (a.channel1 - b.channel1)*(a.channel1 - b.channel1) <= 1
        );
    }

    __global__
    void
    sparse_ccl_kernel(
        const cell * _cells,
        const unsigned int * blocks,
        float * _out
    ) {
        /*
         * Select the relevant output region for this block. Since we are
         * writing 4 floats per block (module) and we have a maximum of
         * MAX_CLUSTERS_PER_MODULE output clusters per block, we need to seek
         * four times that length into the array for each block.
         */
        float * out = &_out[4 * MAX_CLUSTERS_PER_MODULE * blockIdx.x];

        /*
         * Seek the correct cell region in the input data. Again, this is all a
         * contiguous block of memory for now, and we use the blocks array to
         * define the different ranges per block/module. At the end of this we
         * have the starting address of the block of cells dedicated to this
         * module, and we have its size.
         */
        const cell * cells = &_cells[blocks[blockIdx.x]];
        const unsigned int size = blocks[blockIdx.x + 1] - blocks[blockIdx.x];

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
         */
        unsigned int adjv[MAX_ACTIVATIONS_PER_MODULE / THREADS_PER_BLOCK][8];
        unsigned int adjc[MAX_ACTIVATIONS_PER_MODULE / THREADS_PER_BLOCK];

        /*
         * This variable will be used to write to the output later. We declare
         * and set it here because it saves us a synchronisation barrier. Not a
         * huge deal, but worth keeping in mind.
         */
        __shared__ unsigned int outi;

        if (threadIdx.x == 0) {
            outi = 0;
        }

        /*
         * These arrays are the meat of the pudding of this algorithm, and we
         * will constantly be writing and reading from them which is why we
         * declare them to be in the fast shared memory. Note that this places a
         * limit on the maximum activations per module, as the amount of shared
         * memory is limited. These could always be moved to global memory, but
         * the algorithm would be decidedly slower in that case.
         */
        __shared__ unsigned int f[MAX_ACTIVATIONS_PER_MODULE], gf[MAX_ACTIVATIONS_PER_MODULE];

        /*
         * Here we initialize a few variables. This loop pattern is commonly
         * found throughout this code. We iterate over the number of activations
         * each thread must process. Sadly, the CUDA limit is 1024 threads per
         * block and we cannot guarantee that there will be fewer than 1024
         * activations in a module. So each thread must be able to do more than
         * one.
         */
        for (unsigned int tst = 0; tst * blockDim.x + threadIdx.x < size; ++tst) {
            /*
             * This line calculates the activation or cell ID from the thread ID
             * and the value in the outer loop.
             */
            unsigned int tid = tst * blockDim.x + threadIdx.x;

            /*
             * At the start, the values of f and gf should be equal to the ID of
             * the cell.
             */
            f[tid] = tid;
            gf[tid] = tid;

            /*
             * The number of adjacent cells for each cell must start at zero, to
             * avoid uninitialized memory. adjv does not need to be zeroed, as
             * we will only access those values if adjc indicates that the value
             * is set.
             */
            adjc[tst] = 0;
        }

        /*
         * After this is all done, we synchronise the block. I am not absolutely
         * certain that this is necessary here, but the overhead is not that big
         * and we might as well be safe rather than sorry.
         */
        __syncthreads();

        /*
         * This loop initializes the adjacency cache, which essentially
         * translates the sparse CCL problem into a graph CCL problem which we
         * can tackle with well-studied algorithms.
         */
        for (unsigned int tst = 0; tst * blockDim.x + threadIdx.x < size; ++tst) {
            unsigned int tid = tst * blockDim.x + threadIdx.x;

            unsigned int tc0 = cells[tid].channel0;
            unsigned int tc1 = cells[tid].channel1;

            /*
             * First, we traverse the cells backwards, starting from the current
             * cell and working back to the first, collecting adjacent cells
             * along the way.
             */
            for (unsigned int j = tid - 1; j < tid; --j) {
                /*
                 * Since the data is sorted, we can assume that if we see a cell
                 * sufficiently far away in both directions, it becomes
                 * impossible for that cell to ever be adjacent to this one.
                 * This is a small optimisation.
                 */
                if (
                    cells[j].channel0 < tc0 - 1 &&
                    cells[j].channel1 < tc1 - 1
                ) {
                    break;
                }

                /*
                 * If the cell examined is adjacent to the current cell, save it
                 * in the current cell's adjacency set.
                 */
                if (is_adjacent(cells[tid], cells[j])) {
                    adjv[tst][adjc[tst]++] = j;
                }
            }

            /*
             * Now we examine all the cells past the current one, using almost
             * the same logic as in the backwards pass.
             */
            for (unsigned int j = tid + 1; j < size; ++j) {
                /*
                 * Note that this check now looks in the opposite direction! An
                 * important difference.
                 */
                if (
                    cells[j].channel0 > tc0 + 1 &&
                    cells[j].channel1 > tc1 + 1
                ) {
                    break;
                }

                if (is_adjacent(cells[tid], cells[j])) {
                    adjv[tst][adjc[tst]++] = j;
                }
            }
        }

        /*
         * Now that the data has initialized, we synchronize again before we
         * move onto the actual processing part.
         */
        __syncthreads();

        /*
         * The algorithm finishes if an iteration leaves the arrays unchanged.
         * This varible will be set if a change is made, and dictates if another
         * loop is necessary.
         */
        bool gfc;

        do {
            /*
             * Reset the end-parameter to false, so we can set it to true if we
             * make a change to our arrays.
             */
            gfc = false;

            /*
             * The algorithm executes in a loop of three distinct parallel
             * stages. In this first one, we examine adjacent cells and copy
             * their cluster ID if it is lower than our, essentially merging
             * the two together.
             */
            for (unsigned int tst = 0; tst * blockDim.x + threadIdx.x < size; ++tst) {
                unsigned int tid = tst * blockDim.x + threadIdx.x;

                for (unsigned int k = 0; k < adjc[tst]; ++k) {
                    unsigned int j = adjv[tst][k];
                    if (gf[tid] > gf[j]) {
                        f[f[tid]] = gf[j];
                        f[tid] = gf[j];
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
            for (unsigned int tid = threadIdx.x; tid < size; tid += blockDim.x) {
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
            for (unsigned int tid = threadIdx.x; tid < size; tid += blockDim.x) {
                if (gf[tid] != f[f[tid]]) {
                    gf[tid] = f[f[tid]];
                    gfc = true;
                }
            }

            /*
             * Synchronize before we head on to the next iteration, or before we
             * head to the post-processing stage.
             */
            __syncthreads();

            /*
             * To determine whether we need another iteration, we use block
             * voting mechanics. Each thread checks if it has made any changes
             * to the arrays, and votes. If any thread votes true, all threads
             * will return a true value and go to the next iteration. Only if
             * all threads return false will the loop exit.
             */
        } while (__any_sync(__activemask(), gfc));

        /*
         * This is the post-processing stage, where we merge the clusters into a
         * single measurement and write it to the output.
         */
        for (unsigned int tst = 0; tst * blockDim.x + threadIdx.x < size; ++tst) {
            unsigned int tid = tst * blockDim.x + threadIdx.x;

            /*
             * If and only if the value in the work arrays is equal to the index
             * of a cell, that cell is the "parent" of a cluster of cells. If
             * they are not, there is nothing for us to do. Easy!
             */
            if (f[tid] == tid) {
                /*
                 * These variables keep track of the sums of X and Y coordinates
                 * for the final coordinates, the total activation weight, as
                 * well as the sum of squares of positions, which we use to
                 * calculate the variance.
                 */
                float sx = 0.0, sy = 0.0, sw = 0.0;
                float sx2 = 0.0, sy2 = 0.0;

                /*
                 * Now, we iterate over all other cells to check if they belong
                 * to our cluster. Note that we can start at the current index
                 * because no cell is every a child of a cluster owned by a cell
                 * with a higher ID.
                 */
                for (unsigned int j = tid; j < size; j++) {
                    /*
                     * If the value of this cell is equal to our, that means it
                     * is part of our cluster. In that case, we take its values
                     * for position and add them to our accumulators.
                     */
                    if (f[j] == tid) {
                        sx += cells[j].activation * cells[j].channel0;
                        sy += cells[j].activation * cells[j].channel1;
                        sw += cells[j].activation;

                        sx2 += cells[j].activation * cells[j].channel0 * cells[j].channel0;
                        sy2 += cells[j].activation * cells[j].channel1 * cells[j].channel1;
                    }
                }

                /*
                 * If we are a cluster owner, atomically claim a position in the
                 * output array which we can write to.
                 */
                unsigned int id = atomicAdd(&outi, 1);

                /*
                 * Write the average weighted x and y coordinates, as well as
                 * the weighted average square position, to the output array.
                 */
                out[4 * id + 0] = sx / sw;
                out[4 * id + 1] = sy / sw;
                out[4 * id + 2] = sx2 / sw;
                out[4 * id + 3] = sy2 / sw;
            }
        }
    }

    __host__
    void
    sparse_ccl(
        const cell * _cells,
        const unsigned int * blocks,
        float * _out,
        std::size_t modules
    ) {
        sparse_ccl_kernel<<<modules, THREADS_PER_BLOCK>>>(_cells, blocks, _out);

        cudaDeviceSynchronize();
    }
}}}
