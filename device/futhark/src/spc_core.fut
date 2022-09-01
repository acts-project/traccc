-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

-- The implementation corresponds to Algorithm 2 of the following paper:
-- https://epubs.siam.org/doi/pdf/10.1137/1.9781611976137.5

import "linear"
import "types"

def measurements_to_spacepoints [n] [m] (ts: [n](u64, affine3)) (ms: *[m]measurement): *[m]spacepoint =
    map (\x -> transform (filter ((== x.geometry) <-< (.0)) ts |> head |> (.1)) x.position) ms

entry measurements_to_spacepoints_entry [n] [m]
    (tis: [n]u64) (tts: [n]affine3) (mes: [m]u64) (mgs: [m]u64) (mp0s: [m]f32)
    (mp1s: [m]f32) (mv0s: [m]f32) (mv1s: [m]f32): ([m]f32, [m]f32, [m]f32) =
    let ms = (zip4 mes mgs (zip mp0s mp1s) (zip mv0s mv1s)) |>
        map (\(e, g, p, v) -> {event=e, geometry=g, position=p, variance=v}) in
    measurements_to_spacepoints (zip tis tts) ms |> unzip3
