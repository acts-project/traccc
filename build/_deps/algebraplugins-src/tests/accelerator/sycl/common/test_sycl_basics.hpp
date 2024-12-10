/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Test include(s).
#include "execute_host_test.hpp"
#include "execute_sycl_test.hpp"
#include "test_basics_base.hpp"
#include "test_basics_functors.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/sycl/shared_memory_resource.hpp>

// GoogleTest include(s).
#include <gtest/gtest.h>

// SYCL include(s).
#include <sycl/sycl.hpp>

/// Test case class, to be specialised for the different plugins
template <typename T>
class test_sycl_basics : public test_basics_base<T> {

 public:
  /// Constructor, setting up the inputs for all of the tests
  test_sycl_basics() : test_basics_base<T>(m_resource) {}

 protected:
  /// Queue to be used by all of the tests.
  ::sycl::queue m_queue;
  /// Memory resource for all of the tests.
  vecmem::sycl::shared_memory_resource m_resource{&m_queue};
};
TYPED_TEST_SUITE_P(test_sycl_basics);

/// Test for some basic 2D "vector operations"
TYPED_TEST_P(test_sycl_basics, vector_2d_ops) {

  // Don't run the test at double precision, if the SYCL device doesn't
  // support it.
  if ((typeid(typename TypeParam::scalar) == typeid(double)) &&
      (this->m_queue.get_device().has(::sycl::aspect::fp64) == false)) {
    GTEST_SKIP();
  }

  // Run the test on the host, and on the/a device.
  execute_host_test<vector_2d_ops_functor<TypeParam> >(
      this->m_p1->size(), vecmem::get_data(*(this->m_p1)),
      vecmem::get_data(*(this->m_p2)),
      vecmem::get_data(*(this->m_output_host)));
  execute_sycl_test<vector_2d_ops_functor<TypeParam> >(
      this->m_queue, this->m_p1->size(), vecmem::get_data(*(this->m_p1)),
      vecmem::get_data(*(this->m_p2)),
      vecmem::get_data(*(this->m_output_device)));

  // Compare the outputs.
  this->compareOutputs();
}

/// Test for some basic 3D "vector operations"
TYPED_TEST_P(test_sycl_basics, vector_3d_ops) {

  // Don't run the test at double precision, if the SYCL device doesn't
  // support it.
  if ((typeid(typename TypeParam::scalar) == typeid(double)) &&
      (this->m_queue.get_device().has(::sycl::aspect::fp64) == false)) {
    GTEST_SKIP();
  }

  // This test is just not numerically stable at float precision in optimized
  // mode on some backends. :-( (Cough... HIP... cough...)
#ifdef NDEBUG
  if (typeid(typename TypeParam::scalar) == typeid(float)) {
    GTEST_SKIP();
  }
#endif  // NDEBUG

  // Run the test on the host, and on the/a device.
  execute_host_test<vector_3d_ops_functor<TypeParam> >(
      this->m_v1->size(), vecmem::get_data(*(this->m_v1)),
      vecmem::get_data(*(this->m_v2)),
      vecmem::get_data(*(this->m_output_host)));
  execute_sycl_test<vector_3d_ops_functor<TypeParam> >(
      this->m_queue, this->m_v1->size(), vecmem::get_data(*(this->m_v1)),
      vecmem::get_data(*(this->m_v2)),
      vecmem::get_data(*(this->m_output_device)));

  // Compare the outputs.
  this->compareOutputs();
}

/// Test for handling matrices
TYPED_TEST_P(test_sycl_basics, matrix64_ops) {

  // Don't run the test at double precision, if the SYCL device doesn't
  // support it.
  if ((typeid(typename TypeParam::scalar) == typeid(double)) &&
      (this->m_queue.get_device().has(::sycl::aspect::fp64) == false)) {
    GTEST_SKIP();
  }

  // Run the test on the host, and on the/a device.
  execute_host_test<matrix64_ops_functor<TypeParam> >(
      this->m_m1->size(), vecmem::get_data(*(this->m_m1)),
      vecmem::get_data(*(this->m_output_host)));
  execute_sycl_test<matrix64_ops_functor<TypeParam> >(
      this->m_queue, this->m_m1->size(), vecmem::get_data(*(this->m_m1)),
      vecmem::get_data(*(this->m_output_device)));

  // Compare the outputs.
  this->compareOutputs();
}

/// Test for handling matrices
TYPED_TEST_P(test_sycl_basics, matrix22_ops) {

  // Don't run the test at double precision, if the SYCL device doesn't
  // support it.
  if ((typeid(typename TypeParam::scalar) == typeid(double)) &&
      (this->m_queue.get_device().has(::sycl::aspect::fp64) == false)) {
    GTEST_SKIP();
  }

  // Run the test on the host, and on the/a device.
  execute_host_test<matrix22_ops_functor<TypeParam> >(
      this->m_m2->size(), vecmem::get_data(*(this->m_m2)),
      vecmem::get_data(*(this->m_output_host)));
  execute_sycl_test<matrix22_ops_functor<TypeParam> >(
      this->m_queue, this->m_m2->size(), vecmem::get_data(*(this->m_m2)),
      vecmem::get_data(*(this->m_output_device)));

  // Compare the outputs.
  this->compareOutputs();
}

/// Test for some operations with @c transform3
TYPED_TEST_P(test_sycl_basics, transform3) {

  // Don't run the test at double precision, if the SYCL device doesn't
  // support it.
  if ((typeid(typename TypeParam::scalar) == typeid(double)) &&
      (this->m_queue.get_device().has(::sycl::aspect::fp64) == false)) {
    GTEST_SKIP();
  }

  // Run the test on the host, and on the/a device.
  execute_host_test<transform3_ops_functor<TypeParam> >(
      this->m_t1->size(), vecmem::get_data(*(this->m_t1)),
      vecmem::get_data(*(this->m_t2)), vecmem::get_data(*(this->m_t3)),
      vecmem::get_data(*(this->m_v1)), vecmem::get_data(*(this->m_v2)),
      vecmem::get_data(*(this->m_output_host)));
  execute_sycl_test<transform3_ops_functor<TypeParam> >(
      this->m_queue, this->m_t1->size(), vecmem::get_data(*(this->m_t1)),
      vecmem::get_data(*(this->m_t2)), vecmem::get_data(*(this->m_t3)),
      vecmem::get_data(*(this->m_v1)), vecmem::get_data(*(this->m_v2)),
      vecmem::get_data(*(this->m_output_device)));

  // Compare the outputs.
  this->compareOutputs();
}

REGISTER_TYPED_TEST_SUITE_P(test_sycl_basics, vector_2d_ops, vector_3d_ops,
                            matrix64_ops, matrix22_ops, transform3);
