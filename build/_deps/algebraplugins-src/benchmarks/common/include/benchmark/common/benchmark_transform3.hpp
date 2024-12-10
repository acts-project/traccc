/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "benchmark_vector.hpp"

// System include(s)
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace algebra {

template <typename transform3_t>
void fill_random_trf(std::vector<transform3_t> &);

/// Benchmark for vector operations
template <typename transform3_t>
struct transform3_bm : public vector_bm<typename transform3_t::vector3> {
 private:
  using base_type = vector_bm<typename transform3_t::vector3>;

 public:
  /// Prefix for the benchmark name
  inline static const std::string bm_name{"transform3"};

  std::vector<transform3_t> trfs;

  /// No default construction: Cannot prepare data
  transform3_bm() = delete;
  std::string name() const override { return base_type::name + "_" + bm_name; }

  /// Construct from an externally provided configuration @param cfg
  transform3_bm(benchmark_base::configuration cfg) : base_type{cfg} {

    const std::size_t n_data{this->m_cfg.n_samples() + this->m_cfg.n_warmup()};

    trfs.reserve(n_data);

    fill_random_trf(trfs);
  }

  /// Clear state
  virtual ~transform3_bm() { trfs.clear(); }

  /// Benchmark case
  void operator()(::benchmark::State &state) override {

    const std::size_t n_samples{this->m_cfg.n_samples()};
    const std::size_t n_warmup{this->m_cfg.n_warmup()};

    // Spin down before benchmark (Thread zero is counting the clock)
    if (state.thread_index() == 0 && this->m_cfg.do_sleep()) {
      std::this_thread::sleep_for(std::chrono::seconds(this->m_cfg.n_sleep()));
    }

    // Run the benchmark
    for (auto _ : state) {
      // Warm-up
      state.PauseTiming();
      if (this->m_cfg.do_warmup()) {
        for (std::size_t i{0u}; i < n_warmup; ++i) {
          ::benchmark::DoNotOptimize(
              this->trfs[i].vector_to_global(this->a[i]));
          benchmark::ClobberMemory();
        }
      }
      state.ResumeTiming();

      for (std::size_t i{n_warmup}; i < n_samples + n_warmup; ++i) {
        ::benchmark::DoNotOptimize(this->trfs[i].vector_to_global(this->a[i]));
        benchmark::ClobberMemory();
      }
    }
  }
};

}  // namespace algebra
