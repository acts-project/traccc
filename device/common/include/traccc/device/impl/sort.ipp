/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cassert>

namespace traccc::device {
template <TRACCC_CONSTRAINT(std::movable) T>
TRACCC_DEVICE void swap(T& a, T& b) {
    T t = std::move(a);
    a = std::move(b);
    b = std::move(t);
}

template <TRACCC_CONSTRAINT(concepts::thread_id1) T,
          TRACCC_CONSTRAINT(concepts::barrier) B,
          TRACCC_CONSTRAINT(std::movable) K,
          TRACCC_CONSTRAINT(std::strict_weak_order<K, K>) C>
TRACCC_DEVICE void blockOddEvenSort(T& thread_id, B& barrier, K* keys,
                                    uint32_t num_keys, C&& comparison) {
    bool sorted;

    do {
        sorted = true;

        for (uint32_t j = 2 * thread_id.getLocalThreadIdX() + 1;
             j < num_keys - 1; j += 2 * thread_id.getBlockDimX()) {
            if (comparison(keys[j + 1], keys[j])) {
                swap(keys[j + 1], keys[j]);
                sorted = false;
            }
        }

        barrier.blockBarrier();

        for (uint32_t j = 2 * thread_id.getLocalThreadIdX(); j < num_keys - 1;
             j += 2 * thread_id.getBlockDimX()) {
            if (comparison(keys[j + 1], keys[j])) {
                swap(keys[j + 1], keys[j]);
                sorted = false;
            }
        }
    } while (barrier.blockOr(!sorted));
}
}  // namespace traccc::device
