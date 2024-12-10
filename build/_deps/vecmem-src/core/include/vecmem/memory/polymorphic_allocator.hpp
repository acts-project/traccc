/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// This header can only be used in C++17 mode. So "device compilers" that
// can't use C++17, must not see it. Which in practically all cases should
// "just" be a question of code organisation.
#if __cplusplus < 201700L
#error \
    "This header can only be used in C++17 mode. " \
         "Ideally it should only be used by the \"host compiler\"."
#endif  // < C++17

/*
 * The purpose of this file is to provide uniform access (on a source-code
 * level) to the polymorphic_allocator type from the standard library. These are
 * either in the std::pmr namespace or in the std::experimental::pmr namespace
 * depending on the GCC version used, so we try to unify them by aliassing
 * depending on the compiler feature flags.
 */
#if defined(VECMEM_HAVE_PMR_MEMORY_RESOURCE)
#include <memory_resource>

namespace vecmem {
template <typename T>
using polymorphic_allocator = std::pmr::polymorphic_allocator<T>;
}
#elif defined(VECMEM_HAVE_EXPERIMENTAL_PMR_MEMORY_RESOURCE)
#include <experimental/memory_resource>

namespace vecmem {
template <typename T>
using polymorphic_allocator = std::experimental::pmr::polymorphic_allocator<T>;
}
#else
#error "C++17 LFTS V1 (P0220R1) component memory_resource not found!?!"
#endif
