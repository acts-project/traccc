/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if __cpp_concepts >= 201907L
#define TRACCC_CONSTRAINT(...) __VA_ARGS__
#elif defined(TRACCC_ENFORCE_CONCEPTS)
#error \
    "`TRACCC_ENFORCE_CONCEPTS` is set, but concepts are not available. This constitutes a fatal error."
#else
#define TRACCC_CONSTRAINT(...) typename
#endif
