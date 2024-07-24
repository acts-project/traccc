-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

import "linear"
import "edm"
import "zip"

import "measurement_creation"
import "spacepoint_formation"

entry cells_to_measurements [n]
    (es: [n]u64) (gs: [n]u64) (c0s: [n]i64) (c1s: [n]i64) (as: [n]f32):
    ([]u64, []u64, []f32, []f32, []f32, []f32) =
    (zip5 es gs c0s c1s as) |>
    map (\(e, g, c0, c1, a) ->
         {event=e, geometry=g, position=(c0, c1), activation=a}) >->
    cells_to_measurements_impl |>
    map (\(x: Measurement) ->
         (x.event, x.geometry, x.position.0, x.position.1,
          x.variance.0, x.variance.1)) >->
    unzip6

entry measurements_to_spacepoints [n] [n'] [m]
    (tis: [n]u64) (tts: [n']f32) (mes: [m]u64) (mgs: [m]u64) (mp0s: [m]f32)
    (mp1s: [m]f32) (mv0s: [m]f32) (mv1s: [m]f32):
    ([m]u64, [m]f32, [m]f32, [m]f32) =
    let ttsr = unflatten_3d (n' / 16) 4 4 tts :> [n]Affine3
    let ms = (zip4 mes mgs (zip mp0s mp1s) (zip mv0s mv1s)) |>
        map (\(e, g, p, v) -> {event=e, geometry=g, position=p, variance=v}) in
    measurements_to_spacepoints_impl (zip tis ttsr) ms |>
    map (\(x: Spacepoint) ->
         (x.event, x.position.0, x.position.1, x.position.2)) >-> unzip4
