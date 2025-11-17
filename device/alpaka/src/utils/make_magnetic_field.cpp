/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/utils/make_magnetic_field.hpp"

#include "magnetic_field_types.hpp"

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/bfield/magnetic_field_types.hpp"
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include "traccc/cuda/utils/make_magnetic_field.hpp"
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
//
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include "traccc/sycl/make_magnetic_field.hpp"
#endif

// System include(s).
#include <stdexcept>

namespace traccc::alpaka {

magnetic_field make_magnetic_field(const magnetic_field& bfield,
                                   const queue& queue) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    return traccc::cuda::make_magnetic_field(
              bfield, traccc::cuda::magnetic_field_storage::global_memory);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    if (bfield.is<const_bfield_backend_t<traccc::scalar> >()) {
        return magnetic_field{covfie::field<const_bfield_backend_t<scalar> >{
            bfield.as_field<const_bfield_backend_t<traccc::scalar> >()}};
    } else if (bfield.is<host::inhom_bfield_backend_t<traccc::scalar> >()) {
        return magnetic_field{
            covfie::field<inhom_bfield_backend_t<traccc::scalar> >(
      bfield.as_field<host::inhom_bfield_backend_t<traccc::scalar>>())};
    }
    else {
      throw std::invalid_argument(
          "Unsupported storage method chosen for inhomogeneous b-field");
    }
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    ::sycl::queue q(::alpaka::getNativeHandle(details::get_queue(queue)));
    traccc::sycl::queue_wrapper qw{&queue};
    return traccc::sycl::make_magnetic_field(bfield, qw);
#endif
}

}  // namespace traccc::alpaka
