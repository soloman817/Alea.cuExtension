[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities.Arch

open Alea.CUDA
open Alea.CUDA.Utilities


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

        
//        type ArchProps =
//            {
//                abstract member LOG_WARP_THREADS    : int  /// Log of the number of threads per warp
//                abstract member WARP_THREADS        : int  /// Number of threads per warp
//                abstract member LOG_SMEM_BANKS      : int  /// Log of the number of smem banks
//                abstract member SMEM_BANKS          : int  /// The number of smem banks
//                abstract member SMEM_BANK_BYTES     : int  /// Size of smem bank words
//                abstract member SMEM_BYTES          : int  /// Maximum SM shared memory
//                abstract member SMEM_ALLOC_UNIT     : int  /// Smem allocation size in bytes
//                abstract member REGS_BY_BLOCK       : bool /// Whether or not the architecture allocates registers by block (or by warp)
//                abstract member REG_ALLOC_UNIT      : int  /// Number of registers allocated at a time per block (or by warp)
//                abstract member WARP_ALLOC_UNIT     : int  /// Granularity of warps for which registers are allocated
//                abstract member MAX_SM_THREADS      : int  /// Maximum number of threads per SM
//                abstract member MAX_SM_THREADBLOCKS : int  /// Maximum number of thread blocks per SM
//                abstract member MAX_BLOCK_THREADS   =
//                                            512,                      /// Maximum number of thread per thread block
//            MAX_SM_REGISTERS    =
//                                            8 * 1024,                 /// Maximum number of registers per SM
//
//        type ArchProps(SM_ARCH:int) = 
//            
//            LOG_WARP_THREADS    =
//                                            5,                        /// Log of the number of threads per warp
//            WARP_THREADS        =
//                                            1 << LOG_WARP_THREADS,    /// Number of threads per warp
//            LOG_SMEM_BANKS      =
//                                            4,                        /// Log of the number of smem banks
//            SMEM_BANKS          =
//                                            1 << LOG_SMEM_BANKS,      /// The number of smem banks
//            SMEM_BANK_BYTES     =
//                                            4,                        /// Size of smem bank words
//            SMEM_BYTES          =
//                                            16 * 1024,                /// Maximum SM shared memory
//            SMEM_ALLOC_UNIT     =
//                                            512,                      /// Smem allocation size in bytes
//            REGS_BY_BLOCK       =
//                                            true,                     /// Whether or not the architecture allocates registers by block (or by warp)
//            REG_ALLOC_UNIT      =
//                                            256,                      /// Number of registers allocated at a time per block (or by warp)
//            WARP_ALLOC_UNIT     =
//                                            2,                        /// Granularity of warps for which registers are allocated
//            MAX_SM_THREADS      =
//                                            768,                      /// Maximum number of threads per SM
//            MAX_SM_THREADBLOCKS =
//                                            8,                        /// Maximum number of thread blocks per SM
//            MAX_BLOCK_THREADS   =
//                                            512,                      /// Maximum number of thread per thread block
//            MAX_SM_REGISTERS    =
//                                            8 * 1024,                 /// Maximum number of registers per SM
//        };
//    };