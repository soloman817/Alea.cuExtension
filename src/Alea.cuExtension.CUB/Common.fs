[<AutoOpen>]
module Alea.cuExtension.CUB.Common

open Alea.CUDA
open Alea.CUDA.Utilities



type InputIterator<'T> = deviceptr<'T>



type Arch =
    {
        mutable CUDA_ARCH : DeviceArch
        mutable CUB_PTX_VERSION : int
        mutable CUB_LOG_WARP_THREADS : int
        mutable CUB_LOG_SMEM_BANKS : int
        mutable CUB_SMEM_BANK_BYTES : int
        mutable CUB_SMEM_BYTES : int
        mutable CUB_SMEM_ALLOC_UNIT : int
        mutable CUB_REGS_BY_BLOCK : bool
        mutable CUB_REG_ALLOC_UNIT : int
        mutable CUB_WARP_ALLOC_UNIT : int
        mutable CUB_MAX_SM_THREADS : int
        mutable CUB_MAX_SM_BLOCKS : int
        mutable CUB_MAX_BLOCK_THREADS : int
        mutable CUB_MAX_SM_REGISTERS : int
        mutable CUB_SUBSCRIPTION_FACTOR : int
    }

    [<ReflectedDefinition>]
    static member Default =
        let arch = Worker.Default.Device.Arch.Number
        {   CUDA_ARCH = Worker.Default.Device.Arch;
            CUB_PTX_VERSION = arch;
            CUB_LOG_WARP_THREADS = 5;
            CUB_LOG_SMEM_BANKS = if arch >= 200 then 5 else 4;
            CUB_SMEM_BANK_BYTES = 4;
            CUB_SMEM_BYTES = if arch >= 200 then (48 * 1024) else (16 * 1024);
            CUB_SMEM_ALLOC_UNIT = if arch >= 300 then 256 elif arch >= 200 then 128 else 512;
            CUB_REGS_BY_BLOCK = if arch >= 200 then false else true;
            CUB_REG_ALLOC_UNIT = if arch >= 300 then 256 elif arch >= 200 then 64 elif arch >= 120 then 512 else 256;
            CUB_WARP_ALLOC_UNIT = if arch >= 300 then 4 else 2;
            CUB_MAX_SM_THREADS = if arch >= 300 then 2048 elif arch >= 200 then 1536 elif arch >= 120 then 1024 else 786;
            CUB_MAX_SM_BLOCKS = if arch >= 300 then 16 else 8;
            CUB_MAX_BLOCK_THREADS = if arch >= 200 then 1024 else 512;
            CUB_MAX_SM_REGISTERS = if arch >= 300 then (64 * 1024) elif arch >= 200 then (32 * 1024) elif arch >= 120 then (16 * 1024) else (8 * 1024);
            CUB_SUBSCRIPTION_FACTOR = if arch >= 300 then 5 else 3 }

let arch = Arch.Default
let CUDA_ARCH = arch.CUDA_ARCH
let [<ReflectedDefinition>] CUB_PTX_VERSION = arch.CUB_PTX_VERSION
let [<ReflectedDefinition>] CUB_LOG_WARP_THREADS = arch.CUB_LOG_WARP_THREADS
let [<ReflectedDefinition>] CUB_LOG_SMEM_BANKS = arch.CUB_LOG_SMEM_BANKS
let [<ReflectedDefinition>] CUB_SMEM_BANK_BYTES = arch.CUB_SMEM_BANK_BYTES
let [<ReflectedDefinition>] CUB_SMEM_BYTES = arch.CUB_SMEM_BYTES
let [<ReflectedDefinition>] CUB_SMEM_ALLOC_UNIT = arch.CUB_SMEM_ALLOC_UNIT
let [<ReflectedDefinition>] CUB_REGS_BY_BLOCK = arch.CUB_REGS_BY_BLOCK
let [<ReflectedDefinition>] CUB_REG_ALLOC_UNIT = arch.CUB_REG_ALLOC_UNIT
let [<ReflectedDefinition>] CUB_WARP_ALLOC_UNIT = arch.CUB_WARP_ALLOC_UNIT
let [<ReflectedDefinition>] CUB_MAX_SM_THREADS = arch.CUB_MAX_SM_THREADS
let [<ReflectedDefinition>] CUB_MAX_SM_BLOCKS = arch.CUB_MAX_SM_BLOCKS
let [<ReflectedDefinition>] CUB_MAX_BLOCK_THREADS = arch.CUB_MAX_BLOCK_THREADS
let [<ReflectedDefinition>] CUB_MAX_SM_REGISTERS = arch.CUB_MAX_SM_REGISTERS
let [<ReflectedDefinition>] CUB_SUBSCRIPTION_FACTOR = arch.CUB_SUBSCRIPTION_FACTOR
let [<ReflectedDefinition>] CUB_PTX_LOG_WARP_THREADS = CUB_LOG_WARP_THREADS
let [<ReflectedDefinition>] CUB_PTX_WARP_THREADS = (1 <<< CUB_PTX_LOG_WARP_THREADS)
let [<ReflectedDefinition>] CUB_PTX_LOG_SMEM_BANKS = CUB_LOG_SMEM_BANKS
let [<ReflectedDefinition>] CUB_PTX_SMEM_BANKS = (1 <<< CUB_PTX_LOG_SMEM_BANKS)
let [<ReflectedDefinition>] CUB_PTX_SMEM_BANK_BYTES = CUB_SMEM_BANK_BYTES
let [<ReflectedDefinition>] CUB_PTX_SMEM_BYTES = CUB_SMEM_BYTES
let [<ReflectedDefinition>] CUB_PTX_SMEM_ALLOC_UNIT = CUB_SMEM_ALLOC_UNIT
let [<ReflectedDefinition>] CUB_PTX_REGS_BY_BLOCK = CUB_REGS_BY_BLOCK
let [<ReflectedDefinition>] CUB_PTX_REG_ALLOC_UNIT = CUB_REG_ALLOC_UNIT
let [<ReflectedDefinition>] CUB_PTX_WARP_ALLOC_UNIT = CUB_WARP_ALLOC_UNIT
let [<ReflectedDefinition>] CUB_PTX_MAX_SM_THREADS = CUB_MAX_SM_THREADS
let [<ReflectedDefinition>] CUB_PTX_MAX_SM_BLOCKS = CUB_MAX_SM_BLOCKS
let [<ReflectedDefinition>] CUB_PTX_MAX_BLOCK_THREADS = CUB_MAX_BLOCK_THREADS
let [<ReflectedDefinition>] CUB_PTX_MAX_SM_REGISTERS = CUB_MAX_SM_REGISTERS


let privateStorage() = __null()