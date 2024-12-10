/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(nodiscard) >= 201603L
#define VECMEM_NODISCARD [[nodiscard]]
#else
#define VECMEM_NODISCARD
#endif
#else
#define VECMEM_NODISCARD
#endif
