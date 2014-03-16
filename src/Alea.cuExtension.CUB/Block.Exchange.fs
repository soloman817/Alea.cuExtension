[<AutoOpen>]
module Alea.cuExtension.CUB.Block.Exchange

open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities

open Macro
open Ptx

module Template =
    module Host =
        [<AutoOpen>]
        module Params =
            
            type API =
                {
                    BLOCK_THREADS       :   int
                    ITEMS_PER_THREAD    :   int
                    WARP_TIME_SLICING   :   bool
                }
                        
                
                static member Init(block_threads, items_per_thread, warp_time_slicing) =
                    {
                        BLOCK_THREADS       = block_threads
                        ITEMS_PER_THREAD    = items_per_thread
                        WARP_TIME_SLICING   = warp_time_slicing
                    }
                
                static member Init(block_threads, items_per_thread) = API.Init(block_threads, items_per_thread, false)

        [<AutoOpen>]
        module Constants =
            
            type API =
                {
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
                }
                
                static member Init(p:Params.API) =
                    let log_warp_threads            = CUB_PTX_LOG_WARP_THREADS
                    let warp_threads                = 1 <<< log_warp_threads
                    let warps                       = (p.BLOCK_THREADS + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
                    let log_smem_banks              = CUB_PTX_LOG_SMEM_BANKS
                    let smem_banks                  = 1 <<< log_smem_banks
                    let tile_items                  = p.BLOCK_THREADS * p.ITEMS_PER_THREAD
                    let time_slices                 = if p.WARP_TIME_SLICING then warps else 1
                    let time_sliced_threads         = if p.WARP_TIME_SLICING then (p.BLOCK_THREADS, warp_threads) ||> CUB_MIN else p.BLOCK_THREADS
                    let time_sliced_items           = time_sliced_threads * p.ITEMS_PER_THREAD
                    let warp_time_sliced_threads    = (p.BLOCK_THREADS, warp_threads) ||> CUB_MIN
                    let warp_time_sliced_items      = warp_time_sliced_threads * p.ITEMS_PER_THREAD
                    let insert_padding              = ((p.ITEMS_PER_THREAD &&& (p.ITEMS_PER_THREAD - 1)) = 0)
                    let padding_items               = if insert_padding then time_sliced_items >>> log_smem_banks else 0

                    {
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
                    }

        
        type API =
            {
                Params              : Params.API
                Constants           : Constants.API
                SharedMemoryLength  : int                
            }

            static member Init(block_threads, items_per_thread, warp_time_slicing) =
                let p = Params.API.Init(block_threads, items_per_thread, warp_time_slicing)
                let c = Constants.API.Init(p)
                let sml = c.TIME_SLICED_ITEMS + c.PADDING_ITEMS
                { Params = p; Constants = c; SharedMemoryLength = sml }

    module Device =
        module TempStorage = type API<'T> = deviceptr<'T>
        
        let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:Host.API) = __shared__.Array<'T>(h.SharedMemoryLength) |> __array_to_ptr

        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage.API<'T>
                mutable linear_tid      : int
                mutable warp_lane       : int
                mutable warp_id         : int
                mutable warp_offset     : int
            }

                
            [<ReflectedDefinition>] 
            static member Init(h:Host.API) : API<'T> = 
                let c   = h.Constants
                let linear_tid = threadIdx.x
                let warp_id = linear_tid >>> c.LOG_WARP_THREADS
                {
                    temp_storage    = PrivateStorage<'T>(h)
                    linear_tid      = linear_tid
                    warp_lane       = linear_tid &&& (c.WARP_THREADS - 1)
                    warp_id         = linear_tid >>> c.LOG_WARP_THREADS
                    warp_offset     = warp_id * c.WARP_TIME_SLICED_ITEMS
                }

            [<ReflectedDefinition>]
            static member Init(h:Host.API, temp_storage:TempStorage.API<'T>) =
                let c = h.Constants
                let linear_tid = threadIdx.x
                let warp_id = linear_tid >>> c.LOG_WARP_THREADS
                {
                    temp_storage          = temp_storage
                    linear_tid            = linear_tid
                    warp_lane             = linear_tid &&& (c.WARP_THREADS - 1)
                    warp_id               = warp_id
                    warp_offset           = warp_id * c.WARP_TIME_SLICED_ITEMS
                }

            [<ReflectedDefinition>]
            static member Init(h:Host.API, linear_tid:int) =
                let c   = h.Constants
                let warp_id = linear_tid >>> c.LOG_WARP_THREADS
                {
                    temp_storage    = PrivateStorage<'T>(h)
                    linear_tid      = linear_tid
                    warp_lane       = linear_tid &&& (c.WARP_THREADS - 1)
                    warp_id         = warp_id
                    warp_offset     = warp_id * c.WARP_TIME_SLICED_ITEMS
                }

            [<ReflectedDefinition>]
            static member Init(h:Host.API, temp_storage:TempStorage.API<'T>, linear_tid:int) =
                let c   = h.Constants
                let warp_id = linear_tid >>> c.LOG_WARP_THREADS
                {
                    temp_storage    = temp_storage
                    linear_tid      = linear_tid
                    warp_lane       = linear_tid &&& (c.WARP_THREADS - 1)
                    warp_id         = warp_id
                    warp_offset     = warp_id * c.WARP_TIME_SLICED_ITEMS
                }




            
    type _TemplateParams        = Host.Params.API
    type _Constants             = Host.Constants.API
    type _HostApi               = Host.API
    
    type _TempStorage<'T>       = Device.TempStorage.API<'T>
    type _DeviceApi<'T>         = Device.API<'T>
    
    type _FunctionApi<'T> =
        {
            BlockedToStriped        : Function<_DeviceApi<'T> -> deviceptr<'T> -> unit>
            BlockedToWarpStriped    : Function<_DeviceApi<'T> -> deviceptr<'T> -> unit>
            StripedToBlocked        : Function<_DeviceApi<'T> -> deviceptr<'T> -> unit>
            WarpStripedToBlocked    : Function<_DeviceApi<'T> -> deviceptr<'T> -> unit>
//            ScatterToBlocked        : Function<_DeviceApi<'T> -> deviceptr<'T> -> deviceptr<int> -> unit> // need to edit __ptx__shradd
//            ScatterToStriped        : Function<_DeviceApi<'T> -> deviceptr<'T> -> deviceptr<int> -> unit> // need to edit __ptx__shradd
        }


module private BlockedToStriped =
    open Template
    
    let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) =
        let p = h.Params
        let c = h.Constants
        
        
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (linear_tid * p.ITEMS_PER_THREAD) + ITEM
            if c.INSERT_PADDING then 
                item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = ITEM * p.BLOCK_THREADS + linear_tid
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- temp_storage.[item_offset]
        
    

    let [<ReflectedDefinition>] inline WithTimeslicing (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) =
        let p = h.Params
        let c = h.Constants
        
        let temp_items = __local__.Array<'T>(p.ITEMS_PER_THREAD)
                
        for SLICE = 0 to (c.TIME_SLICES - 1) do
            let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
            let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS

            __syncthreads()

            if warp_id = SLICE then
                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (warp_lane * p.ITEMS_PER_THREAD) + ITEM
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let STRIP_OFFSET = ITEM * p.BLOCK_THREADS
                let STRIP_OOB = STRIP_OFFSET + p.BLOCK_THREADS

                if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                    let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        temp_items.[ITEM] <- temp_storage.[item_offset]

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            items.[ITEM] <- temp_items.[ITEM]   



module private BlockedToWarpStriped =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) = 
        let p = h.Params
        let c = h.Constants
        

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = warp_offset + ITEM + (warp_lane * p.ITEMS_PER_THREAD)
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- temp_storage.[item_offset]

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- temp_storage.[item_offset]
        


    let [<ReflectedDefinition>] inline WithTimeslicing (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) =
        let p = h.Params
        let c = h.Constants
        
        for SLICE = 0 to (c.TIME_SLICES - 1) do
            __syncthreads()
                        
            if warp_id = SLICE then
                for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                    let mutable item_offset = ITEM + (warp_lane * p.ITEMS_PER_THREAD)
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    temp_storage.[item_offset] <- items.[ITEM]

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]
        

    
    //let [<ReflectedDefinition>] inline api (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)  (items:deviceptr<'T>) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h temp_storage linear_tid warp_lane warp_id warp_offset else Default h temp_storage linear_tid warp_lane warp_id warp_offset


module private StripedToBlocked =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) = 
        let p = h.Params
        let c = h.Constants
        
        for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
            let mutable item_offset = (ITEM * p.BLOCK_THREADS) + linear_tid
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (linear_tid * p.ITEMS_PER_THREAD) + ITEM
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- temp_storage.[item_offset]
        


    let [<ReflectedDefinition>] inline WithTimeslicing (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) =
        let p = h.Params
        let c = h.Constants
            
        let temp_items = __local__.Array<'T>(p.ITEMS_PER_THREAD)

        for SLICE = 0 to (c.TIME_SLICES - 1) do
            let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
            let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let STRIP_OFFSET = ITEM * p.BLOCK_THREADS
                let STRIP_OOB = STRIP_OFFSET + p.BLOCK_THREADS
                            
                if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                    let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            if warp_id = SLICE then
                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (warp_lane * p.ITEMS_PER_THREAD) + ITEM
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    temp_items.[ITEM] <- temp_storage.[item_offset]

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            items.[ITEM] <- temp_items.[ITEM]
        

    
//    let [<ReflectedDefinition>] inline api (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)  (items:deviceptr<'T>) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h temp_storage linear_tid warp_lane warp_id warp_offset else Default h temp_storage linear_tid warp_lane warp_id warp_offset


module private WarpStripedToBlocked =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) = 
        let p = h.Params
        let c = h.Constants
            
        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            temp_storage.[item_offset] <- items.[ITEM]

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = warp_offset + ITEM + (warp_lane * p.ITEMS_PER_THREAD)
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- temp_storage.[item_offset]
        


    let [<ReflectedDefinition>] inline WithTimeslicing (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) =
        let p = h.Params
        let c = h.Constants
            
        for SLICE = 0 to (c.TIME_SLICES - 1) do
            __syncthreads()

            if warp_id = SLICE then
                for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                    let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    temp_storage.[item_offset] <- items.[ITEM]

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ITEM + (warp_lane * p.ITEMS_PER_THREAD)
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    items.[ITEM] <- temp_storage.[item_offset]
        


//    let [<ReflectedDefinition>] inline api (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)  (items:deviceptr<'T>) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h temp_storage linear_tid warp_lane warp_id warp_offset else Default h temp_storage linear_tid warp_lane warp_id warp_offset


module private ScatterToBlocked =
    open Template
    
    let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let p = h.Params
        let c = h.Constants

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = ranks.[ITEM]
//            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (linear_tid * p.ITEMS_PER_THREAD) + ITEM
//            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            items.[ITEM] <- temp_storage.[item_offset]
        

    let [<ReflectedDefinition>] inline WithTimeslicing (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let p = h.Params
        let c = h.Constants
            
        let temp_items = __local__.Array<'T>(p.ITEMS_PER_THREAD)
        for SLICE = 0 to (c.TIME_SLICES - 1) do
            __syncthreads()

            let SLICE_OFFSET = c.TIME_SLICED_ITEMS * SLICE

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
//                    if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                    temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()


            if warp_id = SLICE then
                for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                    let mutable item_offset = (warp_lane * p.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                    temp_items.[ITEM] <- temp_storage.[item_offset]

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            items.[ITEM] <- temp_items.[ITEM]
        


//    let [<ReflectedDefinition>] inline api (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)  (items:deviceptr<'T>) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h temp_storage linear_tid warp_lane warp_id warp_offset else Default h temp_storage linear_tid warp_lane warp_id warp_offset


module ScatterToStriped =
    open Template

    let [<ReflectedDefinition>] inline Default (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let p = h.Params
        let c = h.Constants
            

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = ranks.[ITEM]
//            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (ITEM * p.BLOCK_THREADS) + linear_tid
//            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            items.[ITEM] <- temp_storage.[item_offset]
        
    

    let [<ReflectedDefinition>] inline WithTimeslicing (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let p = h.Params
        let c = h.Constants
            

        let temp_items = __local__.Array(p.ITEMS_PER_THREAD)
        for SLICE = 0 to (c.TIME_SLICES - 1) do
            let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
            let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
//                    if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                    temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let STRIP_OFFSET = ITEM * p.BLOCK_THREADS
                let STRIP_OOB = STRIP_OFFSET + p.BLOCK_THREADS

                if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                    let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        temp_items.[ITEM] <- temp_storage.[item_offset]

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            items.[ITEM] <- temp_items.[ITEM]
        


//    let [<ReflectedDefinition>] inline api (h:_HostApi) (temp_storage:_TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int)  (items:deviceptr<'T>) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h temp_storage linear_tid warp_lane warp_id warp_offset else Default h temp_storage linear_tid warp_lane warp_id warp_offset


module BlockExchange =
    
    type TemplateParams     = Template._TemplateParams
    type Constants          = Template._Constants
    type TempStorage<'T>    = Template._TempStorage<'T>
    
    type HostApi            = Template._HostApi
    type private DeviceApi<'T>      = Template._DeviceApi<'T>
    type FunctionApi<'T>    = Template._FunctionApi<'T>

    let [<ReflectedDefinition>] inline PrivateStorage<'T>(h:HostApi) = Template.Device.PrivateStorage<'T>(h)
    
    module BlockedToStriped = 
        let [<ReflectedDefinition>] api (h:HostApi) (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) = 
            if h.Params.WARP_TIME_SLICING then  BlockedToStriped.WithTimeslicing    h temp_storage linear_tid warp_lane warp_id warp_offset items
            else                                BlockedToStriped.Default            h temp_storage linear_tid warp_lane warp_id warp_offset items

    module BlockedToWarpStriped = 
        let [<ReflectedDefinition>] api (h:HostApi) (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) = 
            if h.Params.WARP_TIME_SLICING then  BlockedToWarpStriped.WithTimeslicing    h temp_storage linear_tid warp_lane warp_id warp_offset items
            else                                BlockedToWarpStriped.Default            h temp_storage linear_tid warp_lane warp_id warp_offset items

    module StripedToBlocked = 
        let [<ReflectedDefinition>] api (h:HostApi) (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) = 
            if h.Params.WARP_TIME_SLICING then  BlockedToStriped.WithTimeslicing    h temp_storage linear_tid warp_lane warp_id warp_offset items
            else                                BlockedToStriped.Default            h temp_storage linear_tid warp_lane warp_id warp_offset items

    module WarpStripedToBlocked = 
        let [<ReflectedDefinition>] api (h:HostApi) (temp_storage:TempStorage<'T>) (linear_tid:int) (warp_lane:int) (warp_id:int) (warp_offset:int) (items:deviceptr<'T>) = 
            if h.Params.WARP_TIME_SLICING then  BlockedToStriped.WithTimeslicing    h temp_storage linear_tid warp_lane warp_id warp_offset items
            else                                BlockedToStriped.Default            h temp_storage linear_tid warp_lane warp_id warp_offset items
    
    [<Record>]
    type API<'T> =
        {
            mutable temp_storage    : TempStorage<'T>
            mutable linear_tid      : int
            mutable warp_lane       : int
            mutable warp_id         : int
            mutable warp_offset     : int
        }

                
        [<ReflectedDefinition>] 
        static member Init(h:HostApi) : API<'T> = 
            let c   = h.Constants
            let linear_tid = threadIdx.x
            let warp_id = linear_tid >>> c.LOG_WARP_THREADS
            {
                temp_storage    = PrivateStorage<'T>(h)
                linear_tid      = linear_tid
                warp_lane       = linear_tid &&& (c.WARP_THREADS - 1)
                warp_id         = linear_tid >>> c.LOG_WARP_THREADS
                warp_offset     = warp_id * c.WARP_TIME_SLICED_ITEMS
            }

        [<ReflectedDefinition>]
        static member Init(h:HostApi, temp_storage:TempStorage<'T>) =
            let c = h.Constants
            let linear_tid = threadIdx.x
            let warp_id = linear_tid >>> c.LOG_WARP_THREADS
            {
                temp_storage          = temp_storage
                linear_tid            = linear_tid
                warp_lane             = linear_tid &&& (c.WARP_THREADS - 1)
                warp_id               = warp_id
                warp_offset           = warp_id * c.WARP_TIME_SLICED_ITEMS
            }

        [<ReflectedDefinition>]
        static member Init(h:HostApi, linear_tid:int) =
            let c   = h.Constants
            let warp_id = linear_tid >>> c.LOG_WARP_THREADS
            {
                temp_storage    = PrivateStorage<'T>(h)
                linear_tid      = linear_tid
                warp_lane       = linear_tid &&& (c.WARP_THREADS - 1)
                warp_id         = warp_id
                warp_offset     = warp_id * c.WARP_TIME_SLICED_ITEMS
            }

        [<ReflectedDefinition>]
        static member Init(h:HostApi, temp_storage:TempStorage<'T>, linear_tid:int) =
            let c   = h.Constants
            let warp_id = linear_tid >>> c.LOG_WARP_THREADS
            {
                temp_storage    = temp_storage
                linear_tid      = linear_tid
                warp_lane       = linear_tid &&& (c.WARP_THREADS - 1)
                warp_id         = warp_id
                warp_offset     = warp_id * c.WARP_TIME_SLICED_ITEMS
            }


        [<ReflectedDefinition>] member this.BlockedToStriped(h, items)     = BlockedToStriped.api h this.temp_storage this.linear_tid this.warp_lane this.warp_id this.warp_offset items
        [<ReflectedDefinition>] member this.BlockedToWarpStriped(h, items) = BlockedToWarpStriped.api h this.temp_storage this.linear_tid this.warp_lane this.warp_id this.warp_offset items
        [<ReflectedDefinition>] member this.StripedToBlocked(h, items)     = StripedToBlocked.api h this.temp_storage this.linear_tid this.warp_lane this.warp_id this.warp_offset items
        [<ReflectedDefinition>] member this.WarpStripedToBlocked(h, items) = WarpStripedToBlocked.api h this.temp_storage this.linear_tid this.warp_lane this.warp_id this.warp_offset items

///    let template<'T> (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) : Template<HostApi*FunctionApi<'T>> = cuda {
//        let h = HostApi.Init(block_threads, items_per_thread, warp_time_slicing)
//        
//        let! bts    = h |> BlockedToStriped.api     |> Compiler.DefineFunction
//        let! btws   = h |> BlockedToWarpStriped.api |> Compiler.DefineFunction
//        let! stb    = h |> StripedToBlocked.api     |> Compiler.DefineFunction
//        let! wstb   = h |> WarpStripedToBlocked.api |> Compiler.DefineFunction
//
//        return h, {             
//            BlockedToStriped        = bts
//            BlockedToWarpStriped    = btws
//            StripedToBlocked        = stb
//            WarpStripedToBlocked    = wstb            
//            }}
//
//    module BlockedToStriped     = let api (h:HostApi) = BlockedToStriped.api h
//    module BlockedToWarpStriped = let api (h:HostApi) = BlockedToWarpStriped.api h
//    module StripedToBlocked     = let api (h:HostApi) = StripedToBlocked.api h
//    module WarpStripedToBlocked = let api (h:HostApi) = WarpStripedToBlocked.api h