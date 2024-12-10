/** Algebra plugins library, part of the ACTS project
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "algebra/qualifiers.hpp"

// Local include(s).
#include "test_device_basics.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>

/// Base class for all of the functors
template <typename T>
class functor_base {

 protected:
  /// Tester object
  test_device_basics<T> m_tester;
};

/// Functor running @c test_device_basics::vector_2d_ops
template <typename T>
class vector_2d_ops_functor : public functor_base<T> {

 public:
  ALGEBRA_HOST_DEVICE void operator()(
      std::size_t i, vecmem::data::vector_view<const typename T::point2> a,
      vecmem::data::vector_view<const typename T::point2> b,
      vecmem::data::vector_view<typename T::scalar> output) const {

    // Create the VecMem vector(s).
    vecmem::device_vector<const typename T::point2> vec_a(a), vec_b(b);
    vecmem::device_vector<typename T::scalar> vec_output(output);

    // Perform the operation.
    auto ii = static_cast<typename decltype(vec_output)::size_type>(i);
    vec_output[ii] = this->m_tester.vector_2d_ops(vec_a[ii], vec_b[ii]);
  }
};

/// Functor running @c test_device_basics::vector_3d_ops
template <typename T>
class vector_3d_ops_functor : public functor_base<T> {

 public:
  ALGEBRA_HOST_DEVICE void operator()(
      std::size_t i, vecmem::data::vector_view<const typename T::vector3> a,
      vecmem::data::vector_view<const typename T::vector3> b,
      vecmem::data::vector_view<typename T::scalar> output) const {

    // Create the VecMem vector(s).
    vecmem::device_vector<const typename T::vector3> vec_a(a), vec_b(b);
    vecmem::device_vector<typename T::scalar> vec_output(output);

    // Perform the operation.
    auto ii = static_cast<typename decltype(vec_output)::size_type>(i);
    vec_output[ii] = this->m_tester.vector_3d_ops(vec_a[ii], vec_b[ii]);
  }
};

/// Functor running @c test_device_basics::matrix64_ops
template <typename T>
class matrix64_ops_functor : public functor_base<T> {

 public:
  ALGEBRA_HOST_DEVICE void operator()(
      std::size_t i,
      vecmem::data::vector_view<const typename T::template matrix<6, 4> > m,
      vecmem::data::vector_view<typename T::scalar> output) const {

    // Create the VecMem vector(s).
    vecmem::device_vector<const typename T::template matrix<6, 4> > vec_m(m);
    vecmem::device_vector<typename T::scalar> vec_output(output);

    // Perform the operation.
    auto ii = static_cast<typename decltype(vec_output)::size_type>(i);
    vec_output[ii] = this->m_tester.matrix64_ops(vec_m[ii]);
  }
};

/// Functor running @c test_device_basics::matrix22_ops
template <typename T>
class matrix22_ops_functor : public functor_base<T> {

 public:
  ALGEBRA_HOST_DEVICE void operator()(
      std::size_t i,
      vecmem::data::vector_view<const typename T::template matrix<2, 2> > m,
      vecmem::data::vector_view<typename T::scalar> output) const {

    // Create the VecMem vector(s).
    vecmem::device_vector<const typename T::template matrix<2, 2> > vec_m(m);
    vecmem::device_vector<typename T::scalar> vec_output(output);

    // Perform the operation.
    auto ii = static_cast<typename decltype(vec_output)::size_type>(i);
    vec_output[ii] = this->m_tester.matrix22_ops(vec_m[ii]);
  }
};

/// Functor running @c test_device_basics::transform3_ops
template <typename T>
class transform3_ops_functor : public functor_base<T> {

 public:
  ALGEBRA_HOST_DEVICE void operator()(
      std::size_t i, vecmem::data::vector_view<const typename T::vector3> t1,
      vecmem::data::vector_view<const typename T::vector3> t2,
      vecmem::data::vector_view<const typename T::vector3> t3,
      vecmem::data::vector_view<const typename T::vector3> a,
      vecmem::data::vector_view<const typename T::vector3> b,
      vecmem::data::vector_view<typename T::scalar> output) const {

    // Create the VecMem vector(s).
    vecmem::device_vector<const typename T::vector3> vec_t1(t1), vec_t2(t2),
        vec_t3(t3), vec_a(a), vec_b(b);
    vecmem::device_vector<typename T::scalar> vec_output(output);

    // Perform the operation.
    auto ii = static_cast<typename decltype(vec_output)::size_type>(i);
    vec_output[ii] = this->m_tester.transform3_ops(
        vec_t1[ii], vec_t2[ii], vec_t3[ii], vec_a[ii], vec_b[ii]);
  }
};
