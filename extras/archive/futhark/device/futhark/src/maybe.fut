-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

type Maybe 'a = #Just a | #Nothing

def fmap 'a 'b (f: a -> b) (v: Maybe a): Maybe b = match v
    case #Just x -> #Just (f x)
    case #Nothing -> #Nothing

def maybe 'a (d: a) (v: Maybe a): a = match v
    case #Just x -> x
    case #Nothing -> d
