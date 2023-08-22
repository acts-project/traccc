/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/utils.hpp"
#include "traccc/cuda/seeding/experimental/spacepoint_formation.hpp"
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc::cuda::experimental {

template <typename detector_t>
spacepoint_formation<detector_t>::spacepoint_formation(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str)
    : m_mr(mr), m_copy(copy), m_stream(str) {}

template <typename detector_t>
spacepoint_collection_types::buffer
spacepoint_formation<detector_t>::operator()(
    const typename detector_t::detector_view_type& det_view,
    const measurement_collection_types::const_view& measurements) const {}

}  // namespace traccc::cuda::experimental