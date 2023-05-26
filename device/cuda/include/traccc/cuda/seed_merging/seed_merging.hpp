/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/edm/nseed.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include <vecmem/utils/copy.hpp>
#include <vecmem/memory/unique_ptr.hpp>


namespace traccc::cuda {

class seed_merging : public algorithm<std::pair<vecmem::unique_alloc_ptr<nseed<20>[]>, uint32_t>(const seed_collection_types::buffer&)> {
    public:
    seed_merging(const traccc::memory_resource& mr, stream& str);

    output_type operator()(
        const seed_collection_types::buffer&) const override;

    private:
    traccc::memory_resource m_mr;
    stream& m_stream;
};

}  // namespace traccc::cuda
