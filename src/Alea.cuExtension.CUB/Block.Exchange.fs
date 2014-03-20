[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Exchange

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro
open Ptx



module BlockExchange =
    type StaticParam =
        {
            BLOCK_THREADS               : int
            ITEMS_PER_THREAD            : int
            WARP_TIME_SLICING           : bool
            LOG_WARP_THREADS            : int
            WARP_THREADS                : int
            WARPS                       : int
            LOG_SMEM_BANKS              : int
            SMEM_BANKS                  : int
            TILE_ITEMS                  : int
            TIME_SLICES                 : int
            TIME_SLICED_THREADS         : int
            TIME_SLICED_ITEMS           : int
            WARP_TIME_SLICED_THREADS    : int
            WARP_TIME_SLICED_ITEMS      : int
            INSERT_PADDING              : bool
            PADDING_ITEMS               : int
            SharedMemoryLength          : int
        }
                        
        static member Init(block_threads, items_per_thread, warp_time_slicing) =
            let log_warp_threads            = CUB_PTX_LOG_WARP_THREADS
            let warp_threads                = 1 <<< log_warp_threads
            let warps                       = (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
            let log_smem_banks              = CUB_PTX_LOG_SMEM_BANKS
            let smem_banks                  = 1 <<< log_smem_banks
            let tile_items                  = block_threads * items_per_thread
            let time_slices                 = if warp_time_slicing then warps else 1
            let time_sliced_threads         = if warp_time_slicing then (block_threads, warp_threads) ||> CUB_MIN else block_threads
            let time_sliced_items           = time_sliced_threads * items_per_thread
            let warp_time_sliced_threads    = (block_threads, warp_threads) ||> CUB_MIN
            let warp_time_sliced_items      = warp_time_sliced_threads * items_per_thread
            let insert_padding              = ((items_per_thread &&& (items_per_thread - 1)) = 0)
            let padding_items               = if insert_padding then time_sliced_items >>> log_smem_banks else 0

            {
                BLOCK_THREADS               = block_threads
                ITEMS_PER_THREAD            = items_per_thread
                WARP_TIME_SLICING           = warp_time_slicing
                LOG_WARP_THREADS            = log_warp_threads
                WARP_THREADS                = warp_threads
                WARPS                       = warps
                LOG_SMEM_BANKS              = log_smem_banks
                SMEM_BANKS                  = smem_banks
                TILE_ITEMS                  = tile_items
                TIME_SLICES                 = time_slices
                TIME_SLICED_THREADS         = time_sliced_threads
                TIME_SLICED_ITEMS           = time_sliced_items
                WARP_TIME_SLICED_THREADS    = warp_time_sliced_threads
                WARP_TIME_SLICED_ITEMS      = warp_time_sliced_items
                INSERT_PADDING              = insert_padding
                PADDING_ITEMS               = padding_items                
                SharedMemoryLength          = time_sliced_items + padding_items
            }                

               
        static member Init(block_threads, items_per_thread) = StaticParam.Init(block_threads, items_per_thread, false)
        
    type TempStorage<'T> = deviceptr<'T>


    module private BlockedToStriped =
    
        let [<ReflectedDefinition>] inline Default (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) =
            
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = (linear_tid * p.ITEMS_PER_THREAD) + ITEM
                if p.INSERT_PADDING then 
                    item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ITEM * p.BLOCK_THREADS + linear_tid
                if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                items.[ITEM] <- temp_storage.[item_offset]
        
    

        let [<ReflectedDefinition>] inline WithTimeslicing (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) =
            let temp_items = __local__.Array<'T>(p.ITEMS_PER_THREAD)
                
            for SLICE = 0 to (p.TIME_SLICES - 1) do
                let SLICE_OFFSET = SLICE * p.TIME_SLICED_ITEMS
                let SLICE_OOB = SLICE_OFFSET + p.TIME_SLICED_ITEMS

                __syncthreads()

                if warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = (warp_lane * p.ITEMS_PER_THREAD) + ITEM
                        if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                        temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let STRIP_OFFSET = ITEM * p.BLOCK_THREADS
                    let STRIP_OOB = STRIP_OFFSET + p.BLOCK_THREADS

                    if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                        let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < p.TIME_SLICED_ITEMS) then
                            if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                            temp_items.[ITEM] <- temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                items.[ITEM] <- temp_items.[ITEM]   


    module private BlockedToWarpStriped =
        let [<ReflectedDefinition>] inline Default (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) = 
        
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = warp_offset + ITEM + (warp_lane * p.ITEMS_PER_THREAD)
                if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                items.[ITEM] <- temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = warp_offset + (ITEM * p.WARP_TIME_SLICED_THREADS) + warp_lane
                if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                items.[ITEM] <- temp_storage.[item_offset]
        


        let [<ReflectedDefinition>] inline WithTimeslicing (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) =
            
            for SLICE = 0 to (p.TIME_SLICES - 1) do
                __syncthreads()
                        
                if warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                        let mutable item_offset = ITEM + (warp_lane * p.ITEMS_PER_THREAD)
                        if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                        temp_storage.[item_offset] <- items.[ITEM]

                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = (ITEM * p.WARP_TIME_SLICED_THREADS) + warp_lane
                        if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                        items.[ITEM] <- temp_storage.[item_offset]
        

    module private StripedToBlocked =
        
        let [<ReflectedDefinition>] inline Default (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) = 
            
            for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                let mutable item_offset = (ITEM * p.BLOCK_THREADS) + linear_tid
                if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = (linear_tid * p.ITEMS_PER_THREAD) + ITEM
                if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                items.[ITEM] <- temp_storage.[item_offset]
        


        let [<ReflectedDefinition>] inline WithTimeslicing (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) =
            
            let temp_items = __local__.Array<'T>(p.ITEMS_PER_THREAD)

            for SLICE = 0 to (p.TIME_SLICES - 1) do
                let SLICE_OFFSET = SLICE * p.TIME_SLICED_ITEMS
                let SLICE_OOB = SLICE_OFFSET + p.TIME_SLICED_ITEMS

                __syncthreads()

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let STRIP_OFFSET = ITEM * p.BLOCK_THREADS
                    let STRIP_OOB = STRIP_OFFSET + p.BLOCK_THREADS
                            
                    if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                        let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < p.TIME_SLICED_ITEMS) then
                            if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                            temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                if warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = (warp_lane * p.ITEMS_PER_THREAD) + ITEM
                        if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                        temp_items.[ITEM] <- temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                items.[ITEM] <- temp_items.[ITEM]
        

    module private WarpStripedToBlocked =
        
        let [<ReflectedDefinition>] inline Default (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) = 
                    
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = warp_offset + (ITEM * p.WARP_TIME_SLICED_THREADS) + warp_lane
                if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                temp_storage.[item_offset] <- items.[ITEM]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = warp_offset + ITEM + (warp_lane * p.ITEMS_PER_THREAD)
                if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                items.[ITEM] <- temp_storage.[item_offset]
        

        let [<ReflectedDefinition>] inline WithTimeslicing (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) =
            
            for SLICE = 0 to (p.TIME_SLICES - 1) do
                __syncthreads()

                if warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                        let mutable item_offset = (ITEM * p.WARP_TIME_SLICED_THREADS) + warp_lane
                        if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                        temp_storage.[item_offset] <- items.[ITEM]

                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = ITEM + (warp_lane * p.ITEMS_PER_THREAD)
                        if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                        items.[ITEM] <- temp_storage.[item_offset]
        

    module private ScatterToBlocked =
        
        let [<ReflectedDefinition>] inline Default (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) (ranks:deviceptr<int>) =
            
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ranks.[ITEM]
    //            if p.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), p.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = (linear_tid * p.ITEMS_PER_THREAD) + ITEM
    //            if p.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), p.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                items.[ITEM] <- temp_storage.[item_offset]
        

        let [<ReflectedDefinition>] inline WithTimeslicing (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) (ranks:deviceptr<int>) =
            
            let temp_items = __local__.Array<'T>(p.ITEMS_PER_THREAD)
            for SLICE = 0 to (p.TIME_SLICES - 1) do
                __syncthreads()

                let SLICE_OFFSET = p.TIME_SLICED_ITEMS * SLICE

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < p.WARP_TIME_SLICED_ITEMS) then
    //                    if p.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), p.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                        temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()


                if warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                        let mutable item_offset = (warp_lane * p.ITEMS_PER_THREAD) + ITEM
    //                    if p.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), p.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                        temp_items.[ITEM] <- temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                items.[ITEM] <- temp_items.[ITEM]


    module private ScatterToStriped =
        
        let [<ReflectedDefinition>] inline Default (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) 
            (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ranks.[ITEM]
    //            if p.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), p.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = (ITEM * p.BLOCK_THREADS) + linear_tid
    //            if p.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), p.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                items.[ITEM] <- temp_storage.[item_offset]
        
    

        let [<ReflectedDefinition>] inline WithTimeslicing (p:StaticParam)
            (temp_storage:TempStorage<'T>) (linear_tid:int) 
            (warp_lane:int) (warp_id:int) (warp_offset:int)
            (items:deviceptr<'T>) (ranks:deviceptr<int>) =
            
            let temp_items = __local__.Array(p.ITEMS_PER_THREAD)
            for SLICE = 0 to (p.TIME_SLICES - 1) do
                let SLICE_OFFSET = SLICE * p.TIME_SLICED_ITEMS
                let SLICE_OOB = SLICE_OFFSET + p.TIME_SLICED_ITEMS

                __syncthreads()

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < p.WARP_TIME_SLICED_ITEMS) then
    //                    if p.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), p.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                        temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let STRIP_OFFSET = ITEM * p.BLOCK_THREADS
                    let STRIP_OOB = STRIP_OFFSET + p.BLOCK_THREADS

                    if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                        let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < p.TIME_SLICED_ITEMS) then
                            if p.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> p.LOG_SMEM_BANKS)
                            temp_items.[ITEM] <- temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                items.[ITEM] <- temp_items.[ITEM]



    let [<ReflectedDefinition>] inline PrivateStorage<'T>(p:StaticParam) = __shared__.Array<'T>(p.SharedMemoryLength) |> __array_to_ptr

    [<Record>]
    type InstanceParam<'T> =
        {
            mutable temp_storage    : TempStorage<'T>
            mutable linear_tid      : int
            mutable warp_lane       : int
            mutable warp_id         : int
            mutable warp_offset     : int
        }

                
        [<ReflectedDefinition>] 
        static member Init(p:StaticParam) : InstanceParam<'T> = 
            
            let linear_tid = threadIdx.x
            let warp_id = linear_tid >>> p.LOG_WARP_THREADS
            {
                temp_storage    = PrivateStorage<'T>(p)
                linear_tid      = linear_tid
                warp_lane       = linear_tid &&& (p.WARP_THREADS - 1)
                warp_id         = linear_tid >>> p.LOG_WARP_THREADS
                warp_offset     = warp_id * p.WARP_TIME_SLICED_ITEMS
            }

        [<ReflectedDefinition>]
        static member Init(p:StaticParam, temp_storage:TempStorage<'T>) =
            
            let linear_tid = threadIdx.x
            let warp_id = linear_tid >>> p.LOG_WARP_THREADS
            {
                temp_storage          = temp_storage
                linear_tid            = linear_tid
                warp_lane             = linear_tid &&& (p.WARP_THREADS - 1)
                warp_id               = warp_id
                warp_offset           = warp_id * p.WARP_TIME_SLICED_ITEMS
            }

        [<ReflectedDefinition>]
        static member Init(p:StaticParam, linear_tid:int) =
            
            let warp_id = linear_tid >>> p.LOG_WARP_THREADS
            {
                temp_storage    = PrivateStorage<'T>(p)
                linear_tid      = linear_tid
                warp_lane       = linear_tid &&& (p.WARP_THREADS - 1)
                warp_id         = warp_id
                warp_offset     = warp_id * p.WARP_TIME_SLICED_ITEMS
            }

        [<ReflectedDefinition>]
        static member Init(p:StaticParam, temp_storage:TempStorage<'T>, linear_tid:int) =
            let warp_id = linear_tid >>> p.LOG_WARP_THREADS
            {
                temp_storage    = temp_storage
                linear_tid      = linear_tid
                warp_lane       = linear_tid &&& (p.WARP_THREADS - 1)
                warp_id         = warp_id
                warp_offset     = warp_id * p.WARP_TIME_SLICED_ITEMS
            }    

        [<ReflectedDefinition>] 
        member this.BlockedToStriped(p:StaticParam, items:deviceptr<'T>) = 
            if p.WARP_TIME_SLICING then 
                BlockedToStriped.WithTimeslicing p 
                    this.temp_storage this.linear_tid 
                    this.warp_lane this.warp_id this.warp_offset
                    items
            else 
                BlockedToStriped.Default p
                    this.temp_storage this.linear_tid
                    this.warp_lane this.warp_id this.warp_offset
                    items

        [<ReflectedDefinition>]
        member this.BlockedToWarpStriped(p:StaticParam, items:deviceptr<'T>) = 
            if p.WARP_TIME_SLICING then 
                BlockedToWarpStriped.WithTimeslicing p 
                    this.temp_storage this.linear_tid 
                    this.warp_lane this.warp_id this.warp_offset
                    items
            else 
                BlockedToWarpStriped.Default p
                    this.temp_storage this.linear_tid
                    this.warp_lane this.warp_id this.warp_offset
                    items

        [<ReflectedDefinition>]
        member this.StripedToBlocked(p:StaticParam, items:deviceptr<'T>) =
            if p.WARP_TIME_SLICING then 
                StripedToBlocked.WithTimeslicing p 
                    this.temp_storage this.linear_tid 
                    this.warp_lane this.warp_id this.warp_offset
                    items
            else 
                StripedToBlocked.Default p
                    this.temp_storage this.linear_tid
                    this.warp_lane this.warp_id this.warp_offset
                    items

        [<ReflectedDefinition>]
        member this.WarpStripedToBlocked(p:StaticParam, items:deviceptr<'T>) = 
            if p.WARP_TIME_SLICING then 
                WarpStripedToBlocked.WithTimeslicing p 
                    this.temp_storage this.linear_tid 
                    this.warp_lane this.warp_id this.warp_offset
                    items
            else 
                WarpStripedToBlocked.Default p
                    this.temp_storage this.linear_tid
                    this.warp_lane this.warp_id this.warp_offset
                    items

        [<ReflectedDefinition>]
        member this.ScatterToBlocked(p:StaticParam, items:deviceptr<'T>, ranks:deviceptr<int>) =
            if p.WARP_TIME_SLICING then
                ScatterToBlocked.WithTimeslicing p
                    this.temp_storage this.linear_tid
                    this.warp_lane this.warp_id this.warp_offset
                    items ranks
            else
                ScatterToBlocked.Default p
                    this.temp_storage this.linear_tid
                    this.warp_lane this.warp_id this.warp_offset
                    items ranks

        [<ReflectedDefinition>]
        member this.ScatterToStriped(p:StaticParam, items:deviceptr<'T>, ranks:deviceptr<int>) =
            if p.WARP_TIME_SLICING then
                ScatterToStriped.WithTimeslicing p
                    this.temp_storage this.linear_tid
                    this.warp_lane this.warp_id this.warp_offset
                    items ranks
            else
                ScatterToStriped.Default p
                    this.temp_storage this.linear_tid
                    this.warp_lane this.warp_id this.warp_offset
                    items ranks

//    type KernelApi<'T> =
//        {
//            BlockedToStriped        : TempStorage<'T> -> int -> int -> int -> int -> deviceptr<'T> -> unit
//            BlockedToWarpStriped    : TempStorage<'T> -> int -> int -> int -> int -> deviceptr<'T> -> unit
//            StripedToBlocked        : TempStorage<'T> -> int -> int -> int -> int -> deviceptr<'T> -> unit
//            WarpStripedToBlocked    : TempStorage<'T> -> int -> int -> int -> int -> deviceptr<'T> -> unit
//            ScatterToBlocked        : TempStorage<'T> -> int -> int -> int -> int -> deviceptr<'T> -> deviceptr<int> -> unit
//            ScatterToStriped        : TempStorage<'T> -> int -> int -> int -> int -> deviceptr<'T> -> deviceptr<int> -> unit
//        }
//
//        static member Init(p:StaticParam) =
//            {
//                BlockedToStriped =
//                    if p.WARP_TIME_SLICING then BlockedToStriped.WithTimeslicing p else BlockedToStriped.Default p
//                BlockedToWarpStriped = 
//                    if p.WARP_TIME_SLICING then BlockedToWarpStriped.WithTimeslicing p else BlockedToWarpStriped.Default p 
//                StripedToBlocked =
//                    if p.WARP_TIME_SLICING then StripedToBlocked.WithTimeslicing p else StripedToBlocked.Default p
//                WarpStripedToBlocked =
//                    if p.WARP_TIME_SLICING then WarpStripedToBlocked.WithTimeslicing p else WarpStripedToBlocked.Default p
//                ScatterToBlocked =
//                    if p.WARP_TIME_SLICING then ScatterToBlocked.WithTimeslicing p else ScatterToBlocked.Default p
//                ScatterToStriped =
//                    if p.WARP_TIME_SLICING then ScatterToStriped.WithTimeslicing p else ScatterToStriped.Default p
//            }
