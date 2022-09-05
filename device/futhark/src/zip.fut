-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

def unzip6 [n] 'a 'b 'c 'd 'e 'f
    (xs: [n](a, b, c, d, e, f)):
    ([n]a, [n]b, [n]c, [n]d, [n]e, [n]f) =
    let (as, bs, cs, ds, des) = unzip5
        (map (\(a, b, c, d, e, f) -> (a, b, c, d, (e, f))) xs)
    let (es, fs) = unzip des
    in (as, bs, cs, ds, es, fs)

def unzip10 [n] 'a 'b 'c 'd 'e 'f 'g 'h 'i 'j
    (xs: [n](a, b, c, d, e, f, g, h, i, j)):
    ([n]a, [n]b, [n]c, [n]d, [n]e, [n]f, [n]g, [n]h, [n]i, [n]j) =
    let (lhs, rhs) = unzip
        (map (\(a, b, c, d, e, f, g, h, i, j) ->
               ((a, b, c, d, e), (f, g, h, i, j))) xs)
    let (as, bs, cs, ds, es) = unzip5 lhs
    let (fs, gs, hs, js, is) = unzip5 rhs in
    (as, bs, cs, ds, es, fs, gs, hs, js, is)
