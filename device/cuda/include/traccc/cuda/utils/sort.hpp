/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::cuda {
/**
 * @brief Swap two values of arbitrary type.
 *
 * @tparam T The type of values to swap.
 *
 * @param a The first object in the swap (will take the value of b).
 * @param b The second object in the swap (will take the value of a).
 */
template <typename T>
__device__ __forceinline__ void swap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}

/**
 * @brief Perform a block-wide odd-even key sorting.
 *
 * This function performs a sorting operation across the entire block, assuming
 * that all the threads in the block are currently active.
 *
 * @warning The behaviour of this function is ill-defined if any of the threads
 * in the block have exited.
 *
 * @warning This method is efficient for sorting small arrays, preferable in
 * shared memory, but given the O(n^2) worst-case performance this should not
 * be used on larger arrays.
 *
 * @tparam K The type of keys to sort.
 * @tparam C The type of the comparison function.
 *
 * @param keys An array of keys to sort.
 * @param num_keys The number of keys in the array to sort.
 * @param comparison A comparison function.
 */
template <typename K, typename C>
__device__ void blockOddEvenKeySort(K* keys, uint32_t num_keys,
                                    C&& comparison) {
    bool sorted;

    do {
        sorted = true;

        for (uint32_t j = 2 * threadIdx.x + 1; j < num_keys - 1;
             j += 2 * blockDim.x) {
            if (comparison(keys[j + 1], keys[j])) {
                swap(keys[j + 1], keys[j]);
                sorted = false;
            }
        }

        __syncthreads();

        for (uint32_t j = 2 * threadIdx.x; j < num_keys - 1;
             j += 2 * blockDim.x) {
            if (comparison(keys[j + 1], keys[j])) {
                swap(keys[j + 1], keys[j]);
                sorted = false;
            }
        }
    } while (__syncthreads_or(!sorted));
}

/**
 * @brief Perform a warp-wide odd-even key sorting.
 *
 * This function performs a sorting operation across a single warp, assuming
 * that all the threads in the warp are currently active.
 *
 * @warning The behaviour of this function is ill-defined if any of the threads
 * in the warp have exited.
 *
 * @warning This method is efficient for sorting small arrays, preferable in
 * shared memory, but given the O(n^2) worst-case performance this should not
 * be used on larger arrays.
 *
 * @tparam K The type of keys to sort.
 * @tparam C The type of the comparison function.
 *
 * @param keys An array of keys to sort.
 * @param num_keys The number of keys in the array to sort.
 * @param comparison A comparison function.
 */
template <typename K, typename C>
__device__ void warpOddEvenKeySort(K* keys, uint32_t num_keys, C&& comparison) {
    bool sorted;

    do {
        sorted = true;

        for (uint32_t j = 2 * (threadIdx.x % warpSize) + 1; j < num_keys - 1;
             j += 2 * warpSize) {
            if (comparison(keys[j + 1], keys[j])) {
                swap(keys[j + 1], keys[j]);
                sorted = false;
            }
        }

        __syncwarp(0xFFFFFFFF);

        for (uint32_t j = 2 * (threadIdx.x % warpSize); j < num_keys - 1;
             j += 2 * warpSize) {
            if (comparison(keys[j + 1], keys[j])) {
                swap(keys[j + 1], keys[j]);
                sorted = false;
            }
        }
    } while (__any_sync(0xFFFFFFFF, !sorted));
}
}  // namespace traccc::cuda
