/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <seeding/track_params_estimation_helper.hpp>
#include <utils/algorithm.hpp>

namespace traccc {
namespace sycl {

/// track parameter estimation for sycl
struct track_params_estimation
    : public algorithm<host_bound_track_parameters_collection(
          host_seed_container&&)> {
    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    track_params_estimation(vecmem::memory_resource& mr) : m_mr(mr) {}

    /// Callable operator for track_params_esitmation
    ///
    /// @param input_type is the seed container
    ///
    /// @return vector of bound track parameters
    output_type operator()(host_seed_container&& seeds) const override;

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace sycl
}  // namespace traccc