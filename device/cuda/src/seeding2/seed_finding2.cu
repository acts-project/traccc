/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <limits>
#include <memory>
#include <traccc/cuda/seeding2/kernels/kd_tree_kernel.hpp>
#include <traccc/cuda/seeding2/kernels/seed_finding_kernel.hpp>
#include <traccc/cuda/seeding2/seed_finding.hpp>
#include <traccc/cuda/seeding2/types/kd_tree.hpp>
#include <traccc/edm/seed.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/seeding/detail/lin_circle.hpp>
#include <traccc/seeding/detail/seeding_config.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#include <vector>

#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"

namespace traccc::cuda {
seed_finding2::seed_finding2(const seedfinder_config& config,
                             const seedfilter_config& filter_config,
                             const traccc::memory_resource& mr,
                             vecmem::copy& copy, stream& str)
    : m_seedfinder_config(config),
      m_seedfilter_config(filter_config),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

seed_finding2::output_type seed_finding2::operator()(
    const spacepoint_collection_types::const_view& sps) const {

    auto [kd_tree_device, kd_tree_size, internal_sp_device_new] =
        create_kd_tree(m_mr.main, m_copy, sps);

    return run_seeding(m_seedfinder_config, m_seedfilter_config, m_mr.main,
                       m_copy, sps, kd_tree_t(kd_tree_device));
}
}  // namespace traccc::cuda
