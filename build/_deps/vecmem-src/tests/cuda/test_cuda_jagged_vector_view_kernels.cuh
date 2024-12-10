/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_view.hpp"

/// Perform a linear transformation using the received vectors
void linearTransform(const vecmem::data::vector_view<int>& constants,
                     const vecmem::data::jagged_vector_view<int>& input,
                     vecmem::data::jagged_vector_view<int>& output);
