/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <covfie/core/backend/primitive/identity.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/backup.hpp>
#include <covfie/core/backend/transformer/clamp.hpp>
#include <covfie/core/backend/transformer/hilbert.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/morton.hpp>
#include <covfie/core/backend/transformer/nearest_neighbour.hpp>
#include <covfie/core/backend/transformer/shuffle.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

using base_field = covfie::backend::identity<covfie::vector::uint1>;

template class covfie::field<covfie::backend::affine<base_field>>;
template class covfie::field<covfie::backend::backup<base_field>>;
template class covfie::field<covfie::backend::clamp<base_field>>;
template class covfie::field<
    covfie::backend::hilbert<covfie::vector::uint2, base_field>>;
template class covfie::field<
    covfie::backend::morton<covfie::vector::uint2, base_field>>;
template class covfie::field<covfie::backend::nearest_neighbour<base_field>>;
template class covfie::field<
    covfie::backend::shuffle<base_field, std::index_sequence<0>>>;
template class covfie::field<
    covfie::backend::strided<covfie::vector::uint2, base_field>>;
