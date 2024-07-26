-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

import "linear"
import "edm"

def measurements_to_spacepoints_impl [n] [m]
    (ts: [n](u64, Affine3)) (ms: *[m]Measurement): *[m]Spacepoint =
    map (\x ->
        { event=x.event
        , position=transform (
            filter ((== x.geometry) <-< (.0)) ts |> head |> (.1)
            ) x.position
        }
    ) ms
