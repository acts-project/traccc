/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "algebra/vc_soa.hpp"
#include "benchmark/common/benchmark_vector.hpp"
#include "benchmark/vc_soa/data_generator.hpp"

// Benchmark include
#include <benchmark/benchmark.h>

using namespace algebra;

/// Run vector benchmarks
int main(int argc, char** argv) {

  constexpr std::size_t n_samples{160000};

  //
  // Prepare benchmarks
  //
  algebra::benchmark_base::configuration cfg_s{};
  // Reduce the number of samples, since a single SoA struct contains multiple
  // vectors
  cfg_s.n_samples(n_samples / Vc::float_v::Size)
      .n_warmup(static_cast<std::size_t>(0.1 * cfg_s.n_samples()));
  cfg_s.do_sleep(false);

  // For double precision we need more samples (less vectors per SoA)
  algebra::benchmark_base::configuration cfg_d{cfg_s};
  cfg_d.n_samples(n_samples / Vc::double_v::Size)
      .n_warmup(static_cast<std::size_t>(0.1 * cfg_d.n_samples()));

  vector_binaryOP_bm<vc_soa::vector3, float, bench_op::add> v_add_s{cfg_s};
  vector_binaryOP_bm<vc_soa::vector3, float, bench_op::sub> v_sub_s{cfg_s};
  vector_binaryOP_bm<vc_soa::vector3, float, bench_op::dot> v_dot_s{cfg_s};
  vector_binaryOP_bm<vc_soa::vector3, float, bench_op::cross> v_cross_s{cfg_s};
  vector_unaryOP_bm<vc_soa::vector3, float, bench_op::normalize> v_normalize_s{
      cfg_s};

  vector_binaryOP_bm<vc_soa::vector3, double, bench_op::add> v_add_d{cfg_d};
  vector_binaryOP_bm<vc_soa::vector3, double, bench_op::sub> v_sub_d{cfg_d};
  vector_binaryOP_bm<vc_soa::vector3, double, bench_op::dot> v_dot_d{cfg_d};
  vector_binaryOP_bm<vc_soa::vector3, double, bench_op::cross> v_cross_d{cfg_d};
  vector_unaryOP_bm<vc_soa::vector3, double, bench_op::normalize> v_normalize_d{
      cfg_d};

  std::cout << "Algebra-Plugins 'vector' benchmark (Vc SoA)\n"
            << "-------------------------------------------\n\n"
            << "(single)\n"
            << cfg_s << "(double)\n"
            << cfg_d;

  //
  // Register all benchmarks
  //
  ::benchmark::RegisterBenchmark((v_add_s.name() + "_single").c_str(), v_add_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_add_d.name() + "_double").c_str(), v_add_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_sub_s.name() + "_single").c_str(), v_sub_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_sub_d.name() + "_double").c_str(), v_sub_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_dot_s.name() + "_single").c_str(), v_dot_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_dot_d.name() + "_double").c_str(), v_dot_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_cross_s.name() + "_single").c_str(),
                                 v_cross_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_cross_d.name() + "_double").c_str(),
                                 v_cross_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_normalize_s.name() + "_single").c_str(),
                                 v_normalize_s)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();
  ::benchmark::RegisterBenchmark((v_normalize_d.name() + "_double").c_str(),
                                 v_normalize_d)
      ->MeasureProcessCPUTime()
      ->ThreadPerCpu();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
}