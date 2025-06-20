/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "fit_forward_src.cuh"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/detector_type_utils.hpp"

namespace traccc::cuda {
using fitter = fitter_for_t<traccc::telescope_detector::device>;

template void fit_forward<fitter>(const dim3& grid_size, const dim3& block_size,
                                  std::size_t shared_mem_size,
                                  const cudaStream_t& stream,
                                  const typename fitter::config_type cfg,
                                  const device::fit_payload<fitter> payload);

}  // namespace traccc::cuda
