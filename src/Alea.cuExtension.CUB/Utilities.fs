[<AutoOpen>]
module Alea.cuExtension.CUB.Utilities

    module Allocator =
        let f() = "allocator"

    module Arch =
        let f() = "arch"
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


    module Debug =
        let f() = "debug"

    module Device = 
        let f() = "device"

    module Iterator =
        let f() = "iterator"

    module Macro =
        let f() = "macro"

        let CUB_MAX a b = if a > b then a else b
        let CUB_MIN a b = if a < b then a else b
        let CUB_QUOTIENT_FLOOR x y = x / y
        let CUB_QUOTIENT_CEILING x y = (x + y - 1) / y
        let CUB_ROUND_UP_NEAREST x y = ((x + y - 1) / y) * y
        let CUB_ROUND_DOWN_NEAREST x y = (x / y) * y
        //let CUB_TYPE_STRING (x:'T) = typeof<x>
        //#define CUB_CAT_(a, b) a ## b
        //#define CUB_CAT(a, b) CUB_CAT_(a, b)
        //#define CUB_STATIC_ASSERT(cond, msg) typedef int CUB_CAT(cub_static_assert, __LINE__)[(cond) ? 1 : -1]

    module Namespace =
        let f() = "namespace"

    module Ptx =
        let f() = "ptx"

    module Type =
        let f() = "type"

    module Vector =
        let f() = "vector"