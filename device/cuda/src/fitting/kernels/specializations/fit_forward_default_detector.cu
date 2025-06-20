/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "fit_forward_src.cuh"
#include "traccc/fitting/details/kalman_fitting_types.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/bfield.hpp"

namespace traccc::cuda {
using fitter = traccc::details::kalman_fitter_t<
    default_detector::device,
    covfie::field<traccc::const_bfield_backend_t<
        default_detector::device::scalar_type>>::view_t>;

template void fit_forward<fitter>(const dim3& grid_size, const dim3& block_size,
                                  std::size_t shared_mem_size,
                                  const cudaStream_t& stream,
                                  const fitting_config& cfg,
                                  const device::fit_payload<fitter>& payload);

}  // namespace traccc::cuda
