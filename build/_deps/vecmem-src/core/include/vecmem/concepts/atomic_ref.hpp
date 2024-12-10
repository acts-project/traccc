/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#if __cpp_concepts >= 201907L
#include <concepts>

#include "vecmem/memory/memory_order.hpp"
#endif

namespace vecmem::concepts {
#if __cpp_concepts >= 201907L
/**
 * @brief Concept to verify that a type functions as an atomic reference.
 */
template <typename T>
concept atomic_ref = requires {
    typename T::value_type;

    requires requires(const T& r, typename T::value_type v) {
        { r = v }
        ->std::same_as<typename T::value_type>;
    };

    requires requires(const T& r, memory_order o) {
        { r.load(o) }
        ->std::same_as<typename T::value_type>;
    };

    requires requires(const T& r, typename T::value_type v, memory_order o) {
        { r.store(v, o) }
        ->std::same_as<void>;
        { r.exchange(v, o) }
        ->std::same_as<typename T::value_type>;
        { r.fetch_add(v, o) }
        ->std::same_as<typename T::value_type>;
        { r.fetch_sub(v, o) }
        ->std::same_as<typename T::value_type>;
        { r.fetch_and(v, o) }
        ->std::same_as<typename T::value_type>;
        { r.fetch_or(v, o) }
        ->std::same_as<typename T::value_type>;
        { r.fetch_xor(v, o) }
        ->std::same_as<typename T::value_type>;
    };

    requires requires(const T& r, typename T::value_type& e,
                      typename T::value_type d, memory_order o1,
                      memory_order o2) {
        { r.compare_exchange_strong(e, d, o1, o2) }
        ->std::same_as<bool>;
        { r.compare_exchange_strong(e, d, o1) }
        ->std::same_as<bool>;
    };
};
#endif
}  // namespace vecmem::concepts
