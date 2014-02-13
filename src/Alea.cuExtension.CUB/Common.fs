[<AutoOpen>]
module Alea.cuExtension.CUB.Common

open Alea.CUDA
open Alea.CUDA.Utilities


type Locale =
    | Host
    | Device

//let privateStorage() = __null()

type Offset = int

//type IScanOp<'T> =
//    abstract op : Expr<'T -> 'T -> 'T>
//
//
//let inline scan_op (op:'T -> 'T -> 'T) = 
//    { new IScanOp<'T> with
//        member this.op = op }

//
//type long = nativeint
//type ulong = unativeint
//type double = float
//type longlong = int64
//type ulonglong = uint64
//
//
//[<Struct>]
//type short4 =
//    val mutable x : sbyte
//    val mutable y : sbyte
//    val mutable z : sbyte
//    val mutable w : sbyte
//
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w
//
//[<Struct>]
//type ushort4 =
//    val mutable x : byte
//    val mutable y : byte
//    val mutable z : byte
//    val mutable w : byte
//
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w
//
//[<Struct>]
//type int2 =
//    val mutable x : int
//    val mutable y : int
//
//    [<ReflectedDefinition>]
//    new (x, y) = {x = x; y = y}
//    override this.ToString() = sprintf "(%d,%d)" this.x this.y
//
//[<Struct>]
//type uint2 =
//    val mutable x : uint32
//    val mutable y : uint32
//    
//    [<ReflectedDefinition>]
//    new (x, y) = {x = x; y = y}
//    override this.ToString() = sprintf "(%d,%d)" this.x this.y
//
//[<Struct>]
//type long2 =
//    val mutable x : long
//    val mutable y : long
//
//    [<ReflectedDefinition>]
//    new (x, y) = {x = x; y = y}
//    override this.ToString() = sprintf "(%d,%d)" this.x this.y
//
//[<Struct>]
//type ulong2 =
//    val mutable x : ulong
//    val mutable y : ulong
//
//    [<ReflectedDefinition>]
//    new (x, y) = {x = x; y = y}
//    override this.ToString() = sprintf "(%d,%d)" this.x this.y
//
//[<Struct>]
//type float2 =
//    val mutable x : float32
//    val mutable y : float32
//    
//    [<ReflectedDefinition>]
//    new (x, y) = {x = x; y = y}
//    override this.ToString() = sprintf "(%f,%f)" this.x this.y
//
//[<Struct>]
//type int4 =
//    val mutable x : int
//    val mutable y : int
//    val mutable z : int
//    val mutable w : int
//
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w
//
//[<Struct>]
//type uint4 =
//    val mutable x : uint32
//    val mutable y : uint32
//    val mutable z : uint32
//    val mutable w : uint32
//
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w
//
//[<Struct>]
//type float4 =
//    val mutable x : float32
//    val mutable y : float32
//    val mutable z : float32
//    val mutable w : float32
//    
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%f,%f,%f,%f)" this.x this.y this.z this.w
//
//[<Struct>]
//type long4 =
//    val mutable x : long
//    val mutable y : long
//    val mutable z : long
//    val mutable w : long
//
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w
//
//[<Struct>]
//type ulong4 =
//    val mutable x : ulong
//    val mutable y : ulong
//    val mutable z : ulong
//    val mutable w : ulong
//
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w
//
//[<Struct>]
//type longlong2 =
//    val mutable x : longlong
//    val mutable y : longlong
//
//    [<ReflectedDefinition>]
//    new (x, y) = {x = x; y = y}
//    override this.ToString() = sprintf "(%d,%d)" this.x this.y
//
//[<Struct>]
//type ulonglong2 =
//    val mutable x : ulonglong
//    val mutable y : ulonglong
//
//    [<ReflectedDefinition>]
//    new (x, y) = {x = x; y = y}
//    override this.ToString() = sprintf "(%d,%d)" this.x this.y
//
//[<Struct>]
//type double2 =
//    val mutable x : double
//    val mutable y : double
//
//    [<ReflectedDefinition>]
//    new (x, y) = {x = x; y = y}
//    override this.ToString() = sprintf "(%f,%f)" this.x this.y
//
//[<Struct>]
//type longlong4 =
//    val mutable x : longlong
//    val mutable y : longlong
//    val mutable z : longlong
//    val mutable w : longlong
//
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w
//
//[<Struct>]
//type ulonglong4 =
//    val mutable x : ulonglong
//    val mutable y : ulonglong
//    val mutable z : ulonglong
//    val mutable w : ulonglong
//
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%d,%d,%d,%d)" this.x this.y this.z this.w
//
//[<Struct>]
//type double4 =
//    val mutable x : double
//    val mutable y : double
//    val mutable z : double
//    val mutable w : double
//
//    [<ReflectedDefinition>]
//    new (x, y, z, w) = {x = x; y = y; z = z; w = w}
//    override this.ToString() = sprintf "(%f,%f,%f,%f)" this.x this.y this.z this.w


//type InputIterator<'T> = deviceptr<'T>
//type OutputIterator<'T> = deviceptr<'T>
//
//
//
//type Arch =
//    {
//        mutable CUDA_ARCH : DeviceArch
//        mutable CUB_PTX_VERSION : int
//        mutable CUB_LOG_WARP_THREADS : int
//        mutable CUB_LOG_SMEM_BANKS : int
//        mutable CUB_SMEM_BANK_BYTES : int
//        mutable CUB_SMEM_BYTES : int
//        mutable CUB_SMEM_ALLOC_UNIT : int
//        mutable CUB_REGS_BY_BLOCK : bool
//        mutable CUB_REG_ALLOC_UNIT : int
//        mutable CUB_WARP_ALLOC_UNIT : int
//        mutable CUB_MAX_SM_THREADS : int
//        mutable CUB_MAX_SM_BLOCKS : int
//        mutable CUB_MAX_BLOCK_THREADS : int
//        mutable CUB_MAX_SM_REGISTERS : int
//        mutable CUB_SUBSCRIPTION_FACTOR : int
//    }
//
//    [<ReflectedDefinition>]
//    static member Default =
//        let arch = Worker.Default.Device.Arch.Number
//        {   CUDA_ARCH = Worker.Default.Device.Arch;
//            CUB_PTX_VERSION = arch;
//            CUB_LOG_WARP_THREADS = 5;
//            CUB_LOG_SMEM_BANKS = if arch >= 200 then 5 else 4;
//            CUB_SMEM_BANK_BYTES = 4;
//            CUB_SMEM_BYTES = if arch >= 200 then (48 * 1024) else (16 * 1024);
//            CUB_SMEM_ALLOC_UNIT = if arch >= 300 then 256 elif arch >= 200 then 128 else 512;
//            CUB_REGS_BY_BLOCK = if arch >= 200 then false else true;
//            CUB_REG_ALLOC_UNIT = if arch >= 300 then 256 elif arch >= 200 then 64 elif arch >= 120 then 512 else 256;
//            CUB_WARP_ALLOC_UNIT = if arch >= 300 then 4 else 2;
//            CUB_MAX_SM_THREADS = if arch >= 300 then 2048 elif arch >= 200 then 1536 elif arch >= 120 then 1024 else 786;
//            CUB_MAX_SM_BLOCKS = if arch >= 300 then 16 else 8;
//            CUB_MAX_BLOCK_THREADS = if arch >= 200 then 1024 else 512;
//            CUB_MAX_SM_REGISTERS = if arch >= 300 then (64 * 1024) elif arch >= 200 then (32 * 1024) elif arch >= 120 then (16 * 1024) else (8 * 1024);
//            CUB_SUBSCRIPTION_FACTOR = if arch >= 300 then 5 else 3 }
//
//let arch = Arch.Default
//let CUDA_ARCH = arch.CUDA_ARCH
//let [<ReflectedDefinition>] CUB_PTX_VERSION = arch.CUB_PTX_VERSION
//let [<ReflectedDefinition>] CUB_LOG_WARP_THREADS = arch.CUB_LOG_WARP_THREADS
//let [<ReflectedDefinition>] CUB_LOG_SMEM_BANKS = arch.CUB_LOG_SMEM_BANKS
//let [<ReflectedDefinition>] CUB_SMEM_BANK_BYTES = arch.CUB_SMEM_BANK_BYTES
//let [<ReflectedDefinition>] CUB_SMEM_BYTES = arch.CUB_SMEM_BYTES
//let [<ReflectedDefinition>] CUB_SMEM_ALLOC_UNIT = arch.CUB_SMEM_ALLOC_UNIT
//let [<ReflectedDefinition>] CUB_REGS_BY_BLOCK = arch.CUB_REGS_BY_BLOCK
//let [<ReflectedDefinition>] CUB_REG_ALLOC_UNIT = arch.CUB_REG_ALLOC_UNIT
//let [<ReflectedDefinition>] CUB_WARP_ALLOC_UNIT = arch.CUB_WARP_ALLOC_UNIT
//let [<ReflectedDefinition>] CUB_MAX_SM_THREADS = arch.CUB_MAX_SM_THREADS
//let [<ReflectedDefinition>] CUB_MAX_SM_BLOCKS = arch.CUB_MAX_SM_BLOCKS
//let [<ReflectedDefinition>] CUB_MAX_BLOCK_THREADS = arch.CUB_MAX_BLOCK_THREADS
//let [<ReflectedDefinition>] CUB_MAX_SM_REGISTERS = arch.CUB_MAX_SM_REGISTERS
//let [<ReflectedDefinition>] CUB_SUBSCRIPTION_FACTOR = arch.CUB_SUBSCRIPTION_FACTOR
//let [<ReflectedDefinition>] CUB_PTX_LOG_WARP_THREADS = CUB_LOG_WARP_THREADS
//let [<ReflectedDefinition>] CUB_PTX_WARP_THREADS = (1 <<< CUB_PTX_LOG_WARP_THREADS)
//let [<ReflectedDefinition>] CUB_PTX_LOG_SMEM_BANKS = CUB_LOG_SMEM_BANKS
//let [<ReflectedDefinition>] CUB_PTX_SMEM_BANKS = (1 <<< CUB_PTX_LOG_SMEM_BANKS)
//let [<ReflectedDefinition>] CUB_PTX_SMEM_BANK_BYTES = CUB_SMEM_BANK_BYTES
//let [<ReflectedDefinition>] CUB_PTX_SMEM_BYTES = CUB_SMEM_BYTES
//let [<ReflectedDefinition>] CUB_PTX_SMEM_ALLOC_UNIT = CUB_SMEM_ALLOC_UNIT
//let [<ReflectedDefinition>] CUB_PTX_REGS_BY_BLOCK = CUB_REGS_BY_BLOCK
//let [<ReflectedDefinition>] CUB_PTX_REG_ALLOC_UNIT = CUB_REG_ALLOC_UNIT
//let [<ReflectedDefinition>] CUB_PTX_WARP_ALLOC_UNIT = CUB_WARP_ALLOC_UNIT
//let [<ReflectedDefinition>] CUB_PTX_MAX_SM_THREADS = CUB_MAX_SM_THREADS
//let [<ReflectedDefinition>] CUB_PTX_MAX_SM_BLOCKS = CUB_MAX_SM_BLOCKS
//let [<ReflectedDefinition>] CUB_PTX_MAX_BLOCK_THREADS = CUB_MAX_BLOCK_THREADS
//let [<ReflectedDefinition>] CUB_PTX_MAX_SM_REGISTERS = CUB_MAX_SM_REGISTERS













