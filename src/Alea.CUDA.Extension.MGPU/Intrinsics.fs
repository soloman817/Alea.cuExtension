module Alea.CUDA.Extension.MGPU.Intrinsics

// this file maps to intrinsics.cuh. it provides some implementation 
// of some low level device functions.

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension.MGPU.Static
open Alea.CUDA.Extension.MGPU.DeviceUtil


// actrually, the DeviceFunction.__brev(x) doesn't provide host code,
// if you call it from host, you will get exception. we can extend this
// later to add host algorithm, but I leave it now, because for now, it 
// is only called from kernel.
let [<ReflectedDefinition>] brev x = DeviceFunction.__brev(x)

let [<ReflectedDefinition>] clz x =
    let mutable i = 31
    let mutable r = 32
    while i >= 0 do
        if ((1 <<< i) &&& x) = 1 then
            r <- 31 - i
        i <- i - 1
    r


let [<ReflectedDefinition>] findLog2 x roundUp = 
    let mutable a = 31 - clz x
    if roundUp then
        a <- a + (if sIsPow2 x then 0 else 1)
    a

let ulonglong_as_uint2 = 
    <@ fun (x:uint64) ->
        let hb = uint32(x &&& 0xffffffffUL)
        let lb = uint32(x >>> 32)
        uint2(hb,lb) @>



let [<ReflectedDefinition>] popc (x:uint32) = 
    let mutable i = 31
    let mutable r = 31u
    while i <> 0 do
        if (uint32(1 <<< i) &&& uint32(x)) <> 0u then
            r <- uint32(31 - i)
        else
            r <- 32u
        i <- i - 1
    r

let [<ReflectedDefinition>] prmt (a:uint32) (b:uint32) (index:uint32) =
    let mutable result = 0u
    for i = 0 to 3 do        
        let sel : uint32 = 0xfu &&& (index >>> (4 * i))
        let mutable x : uint32 = if (7u &&& sel) > 3u then b else a
        x <- 0xffu &&& (x >>> (8 * (3 &&& int sel)))
        if (8 &&& int sel) <> 0 then x <- if (128u &&& x) <> 0u then 0xffu else 0u
        result <- result ||| (x <<< (8 * i))
    result

let [<ReflectedDefinition>] ffs (x:int) =
    let mutable i = 0
    let mutable r = 0
    while i < 32 do
        if ((1 <<< i) &&& x) <> 0 then r <- r + 1
        i <- i + 1
    r



//MGPU_HOST_DEVICE uint bfi(uint x, uint y, uint bit, uint numBits) {
//	uint result;
//#if __CUDA_ARCH__ >= 200
//	result = bfi_ptx(x, y, bit, numBits);
//#else
//	if(bit + numBits > 32) numBits = 32 - bit;
//	uint mask = ((1<< numBits) - 1)<< bit;
//	result = y & ~mask;
//	result |= mask & (x<< bit);
//#endif
//	return result;
//}
let bfi (x:uint32) (y:uint32) (bit:uint32) (numBits:uint32) =
    let mutable result = 0u
    let mutable numBits = numBits
    if (bit + numBits) > 32u then numBits <- 32u - bit
    let mask = ((1u <<< (int numBits)) - 1u) <<< (int bit)
    result <- y &&& ~~~mask
    result <- result ||| (mask &&& (x <<< (int bit)))
    result



//let prmt =
//    for i = 0 to 4 - 1 do
//        let sel = 0xf &&& (index >>> (4 * i))
//        let mutable x = if ((7 &&& sel) > 3) then b else a
//        if (8 &&& sel) then x <- if (128 &&& x) then 0xff else 0
//        result <- result ||| (x <<< (8 * i))
//    result

