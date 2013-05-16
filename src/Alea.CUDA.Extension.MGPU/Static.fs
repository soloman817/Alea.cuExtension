module Alea.CUDA.Extension.MGPU.Static

// This module maps to static.h file in mgpu code. In mgpu, it uses lot
// of C++ meta programming tech to speed up, so you can see he uses templates
// and calculate the result by compiler, and stores the result by enum.
// F# doesn't have template, but we are the compiler, so we can compute it
// in the compiler runtime (which is same as C++ static compiler time). 
// the prefix "s" means these are static calculation, so I simply keep them.

let sIsPow2 (X:int) = 0 = (X &&& (X - 1))

let rec sLogPow2 (X:int) (roundUp:bool) =
    if X = 0 then 0, 0
    elif X = 1 then 0, 0
    else
        let extra = if sIsPow2(X) then 0 else if roundUp then 1 else 0
        let inner = let inner, _ = sLogPow2 (X / 2) true in inner + 1
        let value = inner + extra
        inner, value
