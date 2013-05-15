module Alea.CUDA.Extension.MGPU.Static

let sIsPow2 (X:int) = 0 = (X &&& (X - 1))

let rec sLogPow2 (X:int) (roundUp:bool) =
    if X = 0 then 0, 0
    elif X = 1 then 0, 0
    else
        let extra = if sIsPow2(X) then 0 else if roundUp then 1 else 0
        let inner = let inner, _ = sLogPow2 (X / 2) true in inner + 1
        let value = inner + extra
        inner, value
