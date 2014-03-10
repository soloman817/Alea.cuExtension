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
            [<Record>]
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
            [<Record>]
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

        [<Record>]
        type API =
            {
                Params              : Params.API
                Constants           : Constants.API
                SharedMemoryLength  : int
            }

            static member Init(block_threads, items_per_thread, warp_time_slicing) =
                let p = Params.API.Init(block_threads, items_per_thread, warp_time_slicing)
                let c = Constants.API.Init(p)
                { Params = p; Constants = c; SharedMemoryLength = c.TIME_SLICED_ITEMS + c.PADDING_ITEMS }

    module Device =
        [<AutoOpen>]
        module TempStorage =
            [<Record>]
            type API<'T> =
                {
                    mutable Ptr     : deviceptr<'T>
                    mutable Length  : int
                }

                member this.Item
                    with    [<ReflectedDefinition>] get (idx:int) = this.Ptr.[idx] 
                    and     [<ReflectedDefinition>] set (idx:int) (v:'T) = this.Ptr.[idx] <- v

                [<ReflectedDefinition>] static member Uninitialized() : API<'T> = { Ptr = __null(); Length = 0}

                [<ReflectedDefinition>]
                static member Uninitialized(length) : API<'T> =
                    let s = __shared__.Array(length)
                    let ptr = s |> __array_to_ptr
                    { Ptr = ptr; Length = length}


        [<AutoOpen>]
        module ThreadFields =
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
                static member Uninitialized() : API<'T> =
                    {
                        temp_storage    = TempStorage.API<'T>.Uninitialized()
                        linear_tid      = 0
                        warp_lane       = 0
                        warp_id         = 0
                        warp_offset     = 0
                    }

                [<ReflectedDefinition>]
                static member Uninitialized(length:int) : API<'T> =
                    {
                        temp_storage    = TempStorage.API<'T>.Uninitialized(length)
                        linear_tid      = 0
                        warp_lane       = 0
                        warp_id         = 0
                        warp_offset     = 0
                    }
        
                [<ReflectedDefinition>]
                static member Init(temp_storage:TempStorage.API<'T>, linear_tid, warp_lane, warp_id, warp_offset) =
                    {
                        temp_storage    = temp_storage
                        linear_tid      = linear_tid
                        warp_lane       = warp_lane
                        warp_id         = warp_id
                        warp_offset     = warp_offset
                    }

                [<ReflectedDefinition>] 
                static member Init(length) = API<'T>.Init(TempStorage.API<'T>.Uninitialized(length), 0, 0, 0, 0)
                [<ReflectedDefinition>] 
                static member Init(temp_storage:TempStorage.API<'T>) = API<'T>.Init(temp_storage, 0,0,0,0)

        [<Record>]
        type API<'T> =
            {
                mutable ThreadFields : ThreadFields.API<'T>
            }

            [<ReflectedDefinition>] static member Init() = { ThreadFields = ThreadFields.API<'T>.Uninitialized() }
            [<ReflectedDefinition>] static member Init(sharedMemoryLength) = { ThreadFields = ThreadFields.API<'T>.Uninitialized(sharedMemoryLength)}


            
    type _TemplateParams        = Host.Params.API
    type _Constants             = Host.Constants.API
    type _HostApi               = Host.API
    
    type _TempStorage<'T>       = Device.TempStorage.API<'T>
    type _ThreadFields<'T>      = Device.ThreadFields.API<'T>
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

    let Default (h:_HostApi) = 
        <@ fun  (d:_DeviceApi<'T>) 
                (items:deviceptr<'T>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields
        
            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = (f.linear_tid * p.ITEMS_PER_THREAD) + ITEM
                if c.INSERT_PADDING then 
                    item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                f.temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ITEM * p.BLOCK_THREADS + f.linear_tid
                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                items.[ITEM] <- f.temp_storage.[item_offset]
        @>
    

    let WithTimeslicing (h:_HostApi) =
        <@ fun  (d:_DeviceApi<'T>)
                (items:deviceptr<'T>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields
        
            let temp_items = __local__.Array<'T>(p.ITEMS_PER_THREAD)
                
            for SLICE = 0 to (c.TIME_SLICES - 1) do
                let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
                let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS

                __syncthreads()

                if f.warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = (f.warp_lane * p.ITEMS_PER_THREAD) + ITEM
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        f.temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let STRIP_OFFSET = ITEM * p.BLOCK_THREADS
                    let STRIP_OOB = STRIP_OFFSET + p.BLOCK_THREADS

                    if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                        let mutable item_offset = STRIP_OFFSET + f.linear_tid - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                            temp_items.[ITEM] <- f.temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                items.[ITEM] <- temp_items.[ITEM]
        @>


    let api (h:_HostApi) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h else Default h



module private BlockedToWarpStriped =
    open Template

    let Default (h:_HostApi) = 
        <@ fun  (d:_DeviceApi<'T>) 
                (items:deviceptr<'T>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields
        

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = f.warp_offset + ITEM + (f.warp_lane * p.ITEMS_PER_THREAD)
                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                items.[ITEM] <- f.temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = f.warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + f.warp_lane
                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                items.[ITEM] <- f.temp_storage.[item_offset]
        @>


    let WithTimeslicing (h:_HostApi) =
        <@ fun  (d:_DeviceApi<'T>)
                (items:deviceptr<'T>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields
        
            for SLICE = 0 to (c.TIME_SLICES - 1) do
                __syncthreads()
                        
                if f.warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                        let mutable item_offset = ITEM + (f.warp_lane * p.ITEMS_PER_THREAD)
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        f.temp_storage.[item_offset] <- items.[ITEM]

                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + f.warp_lane
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        items.[ITEM] <- f.temp_storage.[item_offset]
        @>

    
    let api (h:_HostApi) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h else Default h


module private StripedToBlocked =
    open Template

    let Default (h:_HostApi) = 
        <@ fun  (d:_DeviceApi<'T>) 
                (items:deviceptr<'T>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields
        
            for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                let mutable item_offset = (ITEM * p.BLOCK_THREADS) + f.linear_tid
                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                f.temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = (f.linear_tid * p.ITEMS_PER_THREAD) + ITEM
                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                items.[ITEM] <- f.temp_storage.[item_offset]
        @>


    let WithTimeslicing (h:_HostApi) =
        <@ fun  (d:_DeviceApi<'T>)
                (items:deviceptr<'T>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields

            let temp_items = __local__.Array<'T>(p.ITEMS_PER_THREAD)

            for SLICE = 0 to (c.TIME_SLICES - 1) do
                let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
                let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS

                __syncthreads()

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let STRIP_OFFSET = ITEM * p.BLOCK_THREADS
                    let STRIP_OOB = STRIP_OFFSET + p.BLOCK_THREADS
                            
                    if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                        let mutable item_offset = STRIP_OFFSET + f.linear_tid - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                            f.temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                if f.warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = (f.warp_lane * p.ITEMS_PER_THREAD) + ITEM
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        temp_items.[ITEM] <- f.temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                items.[ITEM] <- temp_items.[ITEM]
        @>

    
    let api (h:_HostApi) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h else Default h


module private WarpStripedToBlocked =
    open Template

    let Default (h:_HostApi) = 
        <@ fun  (d:_DeviceApi<'T>) 
                (items:deviceptr<'T>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = f.warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + f.warp_lane
                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                f.temp_storage.[item_offset] <- items.[ITEM]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = f.warp_offset + ITEM + (f.warp_lane * p.ITEMS_PER_THREAD)
                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                items.[ITEM] <- f.temp_storage.[item_offset]
        @>


    let WithTimeslicing (h:_HostApi) =
        <@ fun  (d:_DeviceApi<'T>)
                (items:deviceptr<'T>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields

            for SLICE = 0 to (c.TIME_SLICES - 1) do
                __syncthreads()

                if f.warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                        let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + f.warp_lane
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        f.temp_storage.[item_offset] <- items.[ITEM]

                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                        let mutable item_offset = ITEM + (f.warp_lane * p.ITEMS_PER_THREAD)
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        items.[ITEM] <- f.temp_storage.[item_offset]
        @>


    let api (h:_HostApi) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h else Default h


module private ScatterToBlocked =
    open Template

    let Default (h:_HostApi) = 
        <@ fun  (d:_DeviceApi<'T>) 
                (items:deviceptr<'T>) (ranks:deviceptr<int>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields


            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ranks.[ITEM]
                if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                f.temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = (f.linear_tid * p.ITEMS_PER_THREAD) + ITEM
                if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                items.[ITEM] <- f.temp_storage.[item_offset]
        @>

    let WithTimeslicing (h:_HostApi) =
        <@ fun  (d:_DeviceApi<'T>)
                (items:deviceptr<'T>) (ranks:deviceptr<int>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields
        
            let temp_items = __local__.Array<'T>(p.ITEMS_PER_THREAD)
            for SLICE = 0 to (c.TIME_SLICES - 1) do
                __syncthreads()

                let SLICE_OFFSET = c.TIME_SLICED_ITEMS * SLICE

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
                        if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                        f.temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()


                if f.warp_id = SLICE then
                    for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
                        let mutable item_offset = (f.warp_lane * p.ITEMS_PER_THREAD) + ITEM
                        if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                        temp_items.[ITEM] <- f.temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                items.[ITEM] <- temp_items.[ITEM]
        @>


    let api (h:_HostApi) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h else Default h


module private ScatterToStriped =
    open Template

    let Default (h:_HostApi) = 
        <@ fun  (d:_DeviceApi<'T>) 
                (items:deviceptr<'T>) (ranks:deviceptr<int>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ranks.[ITEM]
                if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                f.temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = (ITEM * p.BLOCK_THREADS) + f.linear_tid
                if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                items.[ITEM] <- f.temp_storage.[item_offset]
        @>
    

    let WithTimeslicing (h:_HostApi) =
        <@ fun  (d:_DeviceApi<'T>)
                (items:deviceptr<'T>) (ranks:deviceptr<int>) ->
            let p = h.Params
            let c = h.Constants
            let f = d.ThreadFields

            let temp_items = __local__.Array(p.ITEMS_PER_THREAD)
            for SLICE = 0 to (c.TIME_SLICES - 1) do
                let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
                let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS

                __syncthreads()

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
                        if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                        f.temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                    let STRIP_OFFSET = ITEM * p.BLOCK_THREADS
                    let STRIP_OOB = STRIP_OFFSET + p.BLOCK_THREADS

                    if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                        let mutable item_offset = STRIP_OFFSET + f.linear_tid - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                            temp_items.[ITEM] <- f.temp_storage.[item_offset]

            for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
                items.[ITEM] <- temp_items.[ITEM]
        @>


    let api (h:_HostApi) = if h.Params.WARP_TIME_SLICING then WithTimeslicing h else Default h


module BlockExchange =
    
    type TemplateParams     = Template._TemplateParams
    type Constants          = Template._Constants
    type TempStorage<'T>    = Template._TempStorage<'T>
    type ThreadFields<'T>   = Template._ThreadFields<'T>
    
    type HostApi            = Template._HostApi
    type DeviceApi<'T>      = Template._DeviceApi<'T>
    type FunctionApi<'T>    = Template._FunctionApi<'T>

    
    [<Record>]
    type API<'T> =
        {
            host                : HostApi
            mutable device      : DeviceApi<'T>
        }

        [<ReflectedDefinition>] 
        member this.Init() = 
            let c   = this.host.Constants
            let mutable f   = this.device.ThreadFields
            f.linear_tid    <- threadIdx.x
            f.warp_lane     <- f.linear_tid &&& (c.WARP_THREADS - 1)
            f.warp_id       <- f.linear_tid >>> c.LOG_WARP_THREADS
            f.warp_offset   <- f.warp_id * c.WARP_TIME_SLICED_ITEMS

        [<ReflectedDefinition>]
        member this.Init(temp_storage:deviceptr<'T>) =
            let c = this.host.Constants
            let f = this.device.ThreadFields
            f.temp_storage.Ptr      <- temp_storage
            f.linear_tid            <- threadIdx.x
            f.warp_lane             <- f.linear_tid &&& (c.WARP_THREADS - 1)
            f.warp_id               <- f.linear_tid >>> c.LOG_WARP_THREADS
            f.warp_offset           <- f.warp_id * c.WARP_TIME_SLICED_ITEMS

        [<ReflectedDefinition>]
        member this.Init(linear_tid:int) =
            let c   = this.host.Constants
            let f   = this.device.ThreadFields
            f.linear_tid    <- linear_tid
            f.warp_lane     <- linear_tid &&& (c.WARP_THREADS - 1)
            f.warp_id       <- linear_tid >>> c.LOG_WARP_THREADS
            f.warp_offset   <- f.warp_id * c.WARP_TIME_SLICED_ITEMS

        [<ReflectedDefinition>] 
        static member Create(h:HostApi) = { host = h; device = DeviceApi<'T>.Init(h.SharedMemoryLength) }
    


    let template<'T> (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) : Template<HostApi*FunctionApi<'T>> = cuda {
        let h = HostApi.Init(block_threads, items_per_thread, warp_time_slicing)
        
        let! bts    = h |> BlockedToStriped.api |> Compiler.DefineFunction
        let! btws   = h |> BlockedToWarpStriped.api |> Compiler.DefineFunction
        let! stb    = h |> StripedToBlocked.api     |> Compiler.DefineFunction
        let! wstb   = h |> WarpStripedToBlocked.api |> Compiler.DefineFunction

        return h, {             
            BlockedToStriped        = bts
            BlockedToWarpStriped    = btws
            StripedToBlocked        = stb
            WarpStripedToBlocked    = wstb            
            }}
