/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

TEST(KokkosBasic, ParalleFor){
  double* y = static_cast<double*>(std::malloc(N * sizeof(double)));
  double* x = static_cast<double*>(std::malloc(N * sizeof(double)));
  double* z = static_cast<double*>(std::malloc(N * sizeof(double)));

  Kokkos::parallel_for( "y_init", N, KOKKOS_LAMBDA ( int i ) {
    y[i] = 1;
  });

  for(int64_t i = 0; i < N; i++){
    ASSERT_EQ(y[i], 1);
  }

  Kokkos::parallel_for( "x_init", N, KOKKOS_LAMBDA ( int i ) {
    x[i] = 2;
  });

  for(int64_t i = 0; i < N; i++){
    ASSERT_EQ(y[i], 2);
  }

  Kokkos::parallel_for( "sum_x_y", N, KOKKOS_LAMBDA ( int i ) {
    z[i] = y[i] + x[i];
  });

  for(int64_t i = 0; i < N; i++){
    ASSERT_EQ(y[i], 3);
  }
}

TEST(KokkosBasic, ParalleFor){
  double* y = static_cast<double*>(std::malloc(N * sizeof(double)));

  Kokkos::parallel_for( "y_init", N, KOKKOS_LAMBDA ( int64_t i ) {
    y[i] = i;
  });

  for(int64_t i = 0; i < N; i++){
    ASSERT_EQ(y[i], 1);
  }

  int64_t total = 0;
  Kokkos::parallel_reduce( "sum_array", N, KOKKOS_LAMBDA ( int64_t i, int64_t &sum ) {
    sum += y[i];
  }, total);
 
  int64_t real_total = 0;
  for(int64_t i = 0; i < N; i++){
    real_total += y[i];
  }

  ASSERT_EQ(total, real_total);
}
