module Alea.CUDA.Extension.MGPU.Intrinsics

// this file maps to intrinsics.cuh. it provides some implementation 
// of some low level device functions.

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension.MGPU.DeviceUtil

// actrually, the DeviceFunction.__brev(x) doesn't provide host code,
// if you call it from host, you will get exception. we can extend this
// later to add host algorithm, but I leave it now, because for now, it 
// is only called from kernel.
let [<ReflectedDefinition>] brev x = DeviceFunction.__brev(x)


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

