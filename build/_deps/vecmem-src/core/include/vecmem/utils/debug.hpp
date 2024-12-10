/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cstdio>

/// Function name to use for printout operations
#ifndef VECMEM_PRINTF
#if defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)
#define VECMEM_PRINTF VECMEM_SYCL_PRINTF_FUNCTION
#else
#define VECMEM_PRINTF printf
#endif
#endif  // not VECMEM_PRINTF

// Attribute(s) to use on the printed message character string variables
#ifndef VECMEM_MSG_ATTRIBUTES
#ifdef __SYCL_DEVICE_ONLY__
#define VECMEM_MSG_ATTRIBUTES __attribute__((opencl_constant))
#else
#define VECMEM_MSG_ATTRIBUTES
#endif
#endif  // not VECMEM_MSG_ATTRIBUTES

/// Set a default maximum level for the printed messages
#ifndef VECMEM_DEBUG_MSG_LVL
#define VECMEM_DEBUG_MSG_LVL 0
#endif  // not VECMEM_DEBUG_MSG_LVL

/// Set a default length that should be skipped from the front of the file names
#ifndef VECMEM_SOURCE_DIR_LENGTH
#define VECMEM_SOURCE_DIR_LENGTH 0
#endif  // not VECMEM_SOURCE_DIR_LENGTH

/// Helper macro for using the underlying @c printf function
///
/// This intermediate function is really only necessary for SYCL compatibility.
/// Since one needs to very carefully declare the character constant that would
/// be given to the oneAPI printf function.
///
#define __VECMEM_PRINT_MSG(MSG, ...)                    \
    do {                                                \
        const VECMEM_MSG_ATTRIBUTES char __msg[] = MSG; \
        VECMEM_PRINTF(__msg, __VA_ARGS__);              \
    } while (false)

/// Print macro for "level 1" debug messages
#if VECMEM_DEBUG_MSG_LVL >= 1
#define __VECMEM_PRINT_1(MSG, ...) __VECMEM_PRINT_MSG(MSG, __VA_ARGS__)
#else
#define __VECMEM_PRINT_1(MSG, ...)
#endif

/// Print macro for "level 2" debug messages
#if VECMEM_DEBUG_MSG_LVL >= 2
#define __VECMEM_PRINT_2(MSG, ...) __VECMEM_PRINT_MSG(MSG, __VA_ARGS__)
#else
#define __VECMEM_PRINT_2(MSG, ...)
#endif

/// Print macro for "level 3" debug messages
#if VECMEM_DEBUG_MSG_LVL >= 3
#define __VECMEM_PRINT_3(MSG, ...) __VECMEM_PRINT_MSG(MSG, __VA_ARGS__)
#else
#define __VECMEM_PRINT_3(MSG, ...)
#endif

/// Print macro for "level 4" debug messages
#if VECMEM_DEBUG_MSG_LVL >= 4
#define __VECMEM_PRINT_4(MSG, ...) __VECMEM_PRINT_MSG(MSG, __VA_ARGS__)
#else
#define __VECMEM_PRINT_4(MSG, ...)
#endif

/// Print macro for "level 5" debug messages
#if VECMEM_DEBUG_MSG_LVL >= 5
#define __VECMEM_PRINT_5(MSG, ...) __VECMEM_PRINT_MSG(MSG, __VA_ARGS__)
#else
#define __VECMEM_PRINT_5(MSG, ...)
#endif

// Implement the main macro(s) a little differently for MSVC and all other
// compilers/preprocessors. Since the "trick" used for GCC/Clang to allow
// the macro to receive 0 or more arguments just confuses MSVC. And the MSVC
// variadic macro handling can deal with 0 or more arguments out of the box
// anyway.
#if defined(_MSC_VER) && (!defined(__clang__))

/// Helper macro for printing debug messages from "any" code
///
/// Since CUDA, HIP and SYCL all provide "printf style" functions for this, the
/// macro also provides an interface like @c printf canonically does.
///
/// @param LVL The integer message level to use. It must have a value in the
///            [1-5] range.
/// @param MSG The text message to use, before the variadic arguments
///
#define VECMEM_DEBUG_MSG(LVL, MSG, ...)                                  \
    __VECMEM_PRINT_##LVL(                                                \
        "[vecmem] %s:%i " MSG "\n",                                      \
        (static_cast<const char*>(__FILE__) + VECMEM_SOURCE_DIR_LENGTH), \
        __LINE__, __VA_ARGS__)

#else

/// Macro used for handling the case of printing a pure/simple character string
#define __VECMEM_DEBUG_MSG(LVL, MSG, ...)                                \
    __VECMEM_PRINT_##LVL(                                                \
        "[vecmem] %s:%i " MSG "\n%s",                                    \
        (static_cast<const char*>(__FILE__) + VECMEM_SOURCE_DIR_LENGTH), \
        __LINE__, __VA_ARGS__)

/// Helper macro for printing debug messages from "any" code
///
/// Since CUDA, HIP and SYCL all provide "printf style" functions for this, the
/// macro also provides an interface like @c printf canonically does.
///
/// @param LVL The integer message level to use. It must have a value in the
///            [1-5] range.
/// @param ... The "printf style" arguments.
///
#define VECMEM_DEBUG_MSG(LVL, ...) __VECMEM_DEBUG_MSG(LVL, __VA_ARGS__, "")

#endif  // MSC
