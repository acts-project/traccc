/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if __cpp_concepts >= 201907L
#define CONSTRAINT(x) x
#define TRACCC_HAVE_CONCEPTS
#else
#define CONSTRAINT(x) typename
#undef TRACCC_HAVE_CONCEPTS
#endif
