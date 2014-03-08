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
    [<AutoOpen>]
    module Params =
        type API =
            {
                BLOCK_THREADS       :   int
                ITEMS_PER_THREAD    :   int
                WARP_TIME_SLICING   :   bool
            }

            [<ReflectedDefinition>]
            member this.Get = (this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)

            [<ReflectedDefinition>]
            static member Init(block_threads, items_per_thread, warp_time_slicing) =
                {
                    BLOCK_THREADS       = block_threads
                    ITEMS_PER_THREAD    = items_per_thread
                    WARP_TIME_SLICING   = warp_time_slicing
                }

            [<ReflectedDefinition>]
            static member Default(block_threads, items_per_thread) = 
                API.Init(block_threads, items_per_thread, false)
    
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

            [<ReflectedDefinition>]
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

            [<ReflectedDefinition>]
            static member Uninitialized(p:Params.API) =
                let c = Constants.API.Init p
                let length = c.TIME_SLICED_ITEMS + c.PADDING_ITEMS
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
            static member Init(temp_storage:TempStorage.API<'T>, linear_tid, warp_lane, warp_id, warp_offset) =
                {
                    temp_storage    = temp_storage
                    linear_tid      = linear_tid
                    warp_lane       = warp_lane
                    warp_id         = warp_id
                    warp_offset     = warp_offset
                }

            [<ReflectedDefinition>]
            static member Init(p:Params.API) = API<'T>.Init(TempStorage.API<'T>.Uninitialized(p), 0, 0, 0, 0)

    type _TemplateParams        = Params.API
    type _Constants             = Constants.API
    type _TempStorage<'T>       = TempStorage.API<'T>
    type _ThreadFields<'T>      = ThreadFields.API<'T>

    [<Record>]
    type API<'T> =
        {
            mutable Params          : Params.API
            mutable Constants       : Constants.API
            mutable ThreadFields    : ThreadFields.API<'T>
        }

        [<ReflectedDefinition>]
        static member Init(block_threads, items_per_thread, warp_time_slicing) =
            let p = Params.API.Init(block_threads, items_per_thread, warp_time_slicing)
            let c = Constants.API.Init(p)
            let f = ThreadFields.API<'T>.Init(p)
            {
                Params          = p
                Constants       = c
                ThreadFields    = f
            }

        [<ReflectedDefinition>] static member Init(p:Params.API) = API<'T>.Init(p.BLOCK_THREADS, p.ITEMS_PER_THREAD, p.WARP_TIME_SLICING)
        [<ReflectedDefinition>] static member Default(block_threads, items_per_thread) = API<'T>.Init(Params.API.Default(block_threads, items_per_thread))


type _Template<'T> = Template.API<'T>


module BlockedToStriped =
    open Template

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (items:deviceptr<'T>) =
        let p = template.Params
        let c = template.Constants
        let linear_tid = template.ThreadFields.linear_tid
        let temp_storage = template.ThreadFields.temp_storage

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

    let [<ReflectedDefinition>] inline WithTimeslicing (template:_Template<'T>)
        (items:deviceptr<'T>) =
        let c = template.Constants
        let p = template.Params
        let warp_id = template.ThreadFields.warp_id
        let warp_lane = template.ThreadFields.warp_lane
        let temp_storage = template.ThreadFields.temp_storage
        let linear_tid = template.ThreadFields.linear_tid

        let temp_items = __local__.Array(p.ITEMS_PER_THREAD)
                
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


    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.Default = Default this.template
        [<ReflectedDefinition>] member this.WithTimeslicing = WithTimeslicing this.template

        [<ReflectedDefinition>] static member Init(template:_Template<'T>) = { template = template }


module BlockedToWarpStriped =
    open Template

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (items:deviceptr<'T>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = f.warp_offset + ITEM + (f.warp_lane * p.ITEMS_PER_THREAD)
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- f.temp_storage.[item_offset]

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = f.warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + f.warp_lane
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- f.temp_storage.[item_offset]

    let [<ReflectedDefinition>] inline WithTimeslicing (template:_Template<'T>)
        (items:deviceptr<'T>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields
        
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

    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.Default = Default this.template
        [<ReflectedDefinition>] member this.WithTimeslicing = WithTimeslicing this.template

        [<ReflectedDefinition>] static member Init(template:_Template<'T>) = { template = template }


module StripedToBlocked =
    open Template

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (items:deviceptr<'T>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields
        
        for ITEM = 0 to (p.ITEMS_PER_THREAD- 1) do
            let mutable item_offset = (ITEM * p.BLOCK_THREADS) + f.linear_tid
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            f.temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (f.linear_tid * p.ITEMS_PER_THREAD) + ITEM
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- f.temp_storage.[item_offset]
            
    let [<ReflectedDefinition>] inline WithTimeslicing (template:_Template<'T>)
        (items:deviceptr<'T>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields

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

    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.Default = Default this.template
        [<ReflectedDefinition>] member this.WithTimeslicing = WithTimeslicing this.template

        [<ReflectedDefinition>] static member Init(template:_Template<'T>) = { template = template }


module WarpStripedToBlocked =
    open Template

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (items:deviceptr<'T>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = f.warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + f.warp_lane
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            f.temp_storage.[item_offset] <- items.[ITEM]

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = f.warp_offset + ITEM + (f.warp_lane * p.ITEMS_PER_THREAD)
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- f.temp_storage.[item_offset]

    let [<ReflectedDefinition>] inline WithTimeslicing (template:_Template<'T>)
        (items:deviceptr<'T>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields

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

    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.Default = Default this.template
        [<ReflectedDefinition>] member this.WithTimeslicing = WithTimeslicing this.template

        [<ReflectedDefinition>] static member Init(template:_Template<'T>) = { template = template }


module ScatterToBlocked =
    open Template

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields


        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = ranks.[ITEM]
            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            f.temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (f.linear_tid * p.ITEMS_PER_THREAD) + ITEM
            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            items.[ITEM] <- f.temp_storage.[item_offset]

    let [<ReflectedDefinition>] inline WithTimeslicing (template:_Template<'T>)
        (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields
        
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

    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.Default = Default this.template
        [<ReflectedDefinition>] member this.WithTimeslicing = WithTimeslicing this.template

        [<ReflectedDefinition>] static member Init(template:_Template<'T>) = { template = template }


module ScatterToStriped =
    open Template

    let [<ReflectedDefinition>] inline Default (template:_Template<'T>)
        (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = ranks.[ITEM]
            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            f.temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (ITEM * p.BLOCK_THREADS) + f.linear_tid
            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            items.[ITEM] <- f.temp_storage.[item_offset]

    let [<ReflectedDefinition>] inline WithTimeslicing (template:_Template<'T>)
        (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let p = template.Params
        let c = template.Constants
        let f = template.ThreadFields

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

    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.Default = Default this.template
        [<ReflectedDefinition>] member this.WithTimeslicing = WithTimeslicing this.template

        [<ReflectedDefinition>] static member Init(template:_Template<'T>) = { template = template }


module BlockExchange =
    open Template

    [<Record>]
    type API<'T> =
        {
            template : _Template<'T>
        }

        [<ReflectedDefinition>] member this.BlockToStriped          = BlockedToStriped.API<'T>.Init(this.template)
        [<ReflectedDefinition>] member this.BlockToWarpStriped      = BlockedToWarpStriped.API<'T>.Init(this.template)
        [<ReflectedDefinition>] member this.StripedToBlocked        = StripedToBlocked.API<'T>.Init(this.template)
        [<ReflectedDefinition>] member this.WarpStripedToBlocked    = WarpStripedToBlocked.API<'T>.Init(this.template)
        [<ReflectedDefinition>] member this.ScatterToBlocked        = ScatterToBlocked.API<'T>.Init(this.template)
        [<ReflectedDefinition>] member this.ScatterToStriped        = ScatterToStriped.API<'T>.Init(this.template)

        [<ReflectedDefinition>] static member Init(template:_Template<'T>) = { template = template }
        [<ReflectedDefinition>] 
        static member Init(block_threads, items_per_thread, warp_time_slicing) =
            API<'T>.Init(_Template<'T>.Init(block_threads, items_per_thread, warp_time_slicing))

//        [<ReflectedDefinition>]
//        static member Create(block_threads, p.ITEMS_PER_THREAD, warp_time_slicing) =
//            let template = _TemplateParams<'T>.Init(block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)
//            let tf = _ThreadFields<'T>.Init(tp)
//            {
//                BlockedToStriped        =   BlockedToStriped.api template tf
//                BlockedToWarpStriped    =   BlockedToWarpStriped.api template tf
//                StripedToBlocked        =   StripedToBlocked.api template tf
//                WarpStripedToBlocked    =   WarpStripedToBlocked.api template tf                                                            
//                ScatterToBlocked        =   ScatterToBlocked.api template tf
//                ScatterToStriped        =   ScatterToStriped.api template tf
//            }
//
//    let [<ReflectedDefinition>] api (template:_Template<'T>)
//         =
//            {
//                BlockedToStriped        =   BlockedToStriped.api template tf
//                BlockedToWarpStriped    =   BlockedToWarpStriped.api template tf
//                StripedToBlocked        =   StripedToBlocked.api template tf
//                WarpStripedToBlocked    =   WarpStripedToBlocked.api template tf                                                            
//                ScatterToBlocked        =   ScatterToBlocked.api template tf
//                ScatterToStriped        =   ScatterToStriped.api template tf
//            }

//let linear_tid = 
//    fun (tidx:int option) ->
//        match tidx with
//        | Some tidx -> tidx
//        | None -> threadIdx.x
//
//let warp_id = 
//    fun (linear_tid:int option) ->
//        match linear_tid with
//        | Some linear_tid -> 
//            linear_tid >>> LOG_WARP_THREADS
//        | None ->
//            threadIdx.x >>> LOG_WARP_THREADS
//
//let warp_lane = 
//    fun (linear_tid:int option) ->
//        match linear_tid with
//        | Some linear_tid ->
//            linear_tid &&& (WARP_THREADS - 1)
//        | None ->
//            threadIdx.x &&& (WARP_THREADS - 1)
//
//let warp_offset = 
//    fun block_threads p.ITEMS_PER_THREAD -> 
//        fun (linear_tid:int option) ->
//            match linear_tid with
//            | Some linear_tid ->
//                (linear_tid |> Some |> warp_id) * ((block_threads, p.ITEMS_PER_THREAD, ()) |||> WARP_TIME_SLICED_ITEMS)
//            | None ->
//                (threadIdx.x |> Some |> warp_id) * ((block_threads, p.ITEMS_PER_THREAD, ()) |||> WARP_TIME_SLICED_ITEMS)



//let blockedToStriped (block_threads:int) (p.ITEMS_PER_THREAD:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING      = ((),p.ITEMS_PER_THREAD,()) |||> INSERT_PADDING
//    let TIME_SLICED_ITEMS   = (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing) |||> TIME_SLICED_ITEMS
//    let TIME_SLICES         = (block_threads, (), warp_time_slicing) |||> TIME_SLICES
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * p.ITEMS_PER_THREAD) + ITEM
//                    if INSERT_PADDING then 
//                        item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ITEM * block_threads + linear_tid
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let temp_items = __local__.Array(p.ITEMS_PER_THREAD)
//            fun (items:deviceptr<int>) ->
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    let warp_id = linear_tid |> Some |> warp_id
//                    let warp_lane = linear_tid |> Some |> warp_lane
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * p.ITEMS_PER_THREAD) + ITEM
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * block_threads
//                        let STRIP_OOB = STRIP_OFFSET + block_threads
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
//                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
        


//let blockedToWarpStriped (block_threads:int) (p.ITEMS_PER_THREAD:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING              = ((), p.ITEMS_PER_THREAD, ())                |||> INSERT_PADDING
//    let WARP_TIME_SLICED_THREADS    = (block_threads, (), ())                   |||> WARP_TIME_SLICED_THREADS
//    let TIME_SLICES                 = (block_threads, (), warp_time_slicing)    |||> TIME_SLICES
//
//    let warp_offset = (block_threads, p.ITEMS_PER_THREAD) ||> warp_offset
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_offset = linear_tid |> Some |> warp_offset
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * p.ITEMS_PER_THREAD)
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_id = linear_tid |> Some |> warp_id
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    __syncthreads()
//                        
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * p.ITEMS_PER_THREAD)
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
        
//
//let stripedToBlocked (block_threads:int) (p.ITEMS_PER_THREAD:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING      = ((), p.ITEMS_PER_THREAD, ())                              |||> INSERT_PADDING
//    let TIME_SLICES         = (block_threads, (), warp_time_slicing)                  |||> TIME_SLICES
//    let TIME_SLICED_ITEMS   = (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)    |||> TIME_SLICED_ITEMS
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * block_threads) + linear_tid
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * p.ITEMS_PER_THREAD) + ITEM
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let temp_items = __local__.Array(p.ITEMS_PER_THREAD)
//
//            fun (items:deviceptr<int>) ->
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * block_threads
//                        let STRIP_OOB = STRIP_OFFSET + block_threads
//                            
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
//                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    let warp_id = linear_tid |> Some |> warp_id
//                    let warp_lane = linear_tid |> Some |> warp_lane
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * p.ITEMS_PER_THREAD) + ITEM
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        

//let warpStripedToBlocked (block_threads:int) (p.ITEMS_PER_THREAD:int) (warp_time_slicing:bool) =
//    let WARP_TIME_SLICED_THREADS    = (block_threads, (), ())                   |||> WARP_TIME_SLICED_THREADS
//    let INSERT_PADDING              = ((), p.ITEMS_PER_THREAD, ())                |||> INSERT_PADDING
//    let TIME_SLICES                 = (block_threads, (), warp_time_slicing)    |||> TIME_SLICES
//    let warp_offset                 = (block_threads, p.ITEMS_PER_THREAD)          ||> warp_offset
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_offset = linear_tid |> Some |> warp_offset
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * p.ITEMS_PER_THREAD)
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_id = linear_tid |> Some |> warp_id
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * p.ITEMS_PER_THREAD)
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
        
//
//let scatterToBlocked (block_threads:int) (p.ITEMS_PER_THREAD:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING          = ((), p.ITEMS_PER_THREAD, ())                            |||> INSERT_PADDING
//    let TIME_SLICES             = (block_threads, (), warp_time_slicing)                |||> TIME_SLICES
//    let TIME_SLICED_ITEMS       = (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)  |||> TIME_SLICED_ITEMS
//    let WARP_TIME_SLICED_ITEMS  = (block_threads, p.ITEMS_PER_THREAD, ())                 |||> WARP_TIME_SLICED_ITEMS
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * p.ITEMS_PER_THREAD) + ITEM
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
//                let temp_items = __local__.Array(p.ITEMS_PER_THREAD)
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    let SLICE_OFFSET = TIME_SLICED_ITEMS * SLICE
//
//                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS) then
//                            if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    let warp_id = linear_tid |> Some |> warp_id
//                    let warp_lane = linear_tid |> Some |> warp_lane
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * p.ITEMS_PER_THREAD) + ITEM
//                            if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]


//let scatterToStriped (block_threads:int) (p.ITEMS_PER_THREAD:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING          = ((), block_threads, ())                               |||> INSERT_PADDING
//    let TIME_SLICES             = (block_threads, (), warp_time_slicing)                |||> TIME_SLICES
//    let TIME_SLICED_ITEMS       = (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)  |||> TIME_SLICED_ITEMS
//    let WARP_TIME_SLICED_ITEMS  = (block_threads, p.ITEMS_PER_THREAD, ())                 |||> WARP_TIME_SLICED_ITEMS
//    
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) (is_valid:deviceptr<int>) (valid_items:int) ->    
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * block_threads) + linear_tid
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let temp_items = __local__.Array(p.ITEMS_PER_THREAD)
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) (is_valid:deviceptr<int>) (valid_items:int) ->    
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS) then
//                            if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * block_threads
//                        let STRIP_OOB = STRIP_OFFSET + block_threads
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
//                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (p.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]

//let vars (temp_storage:deviceptr<int> option) (linear_tid:int option) =
//    match temp_storage, linear_tid with
//    | Some temp_storage, Some linear_tid -> temp_storage,       linear_tid
//    | None,              Some linear_tid -> privateStorage(),   linear_tid
//    | Some temp_storage, None ->            temp_storage,       threadIdx.x
//    | None,              None ->            privateStorage(),   threadIdx.x

//let blockExchange (block_threads:int) (p.ITEMS_PER_THREAD:int) (warp_time_slicing:bool) =
//    fun (temp_storage:deviceptr<int> option) (linear_tid:int option) ->
//        let temp_storage, linear_tid = (temp_storage, linear_tid) ||> vars

//            
//
//[<Record>]
//type Constants =
//    {
//        LOG_WARP_THREADS            : int
//        WARP_THREADS                : int
//        WARPS                       : int
//        c.LOG_SMEM_BANKS              : int
//        SMEM_BANKS                  : int
//        TILE_ITEMS                  : int
//        TIME_SLICES                 : int
//        TIME_SLICED_THREADS         : int
//        TIME_SLICED_ITEMS           : int
//        WARP_TIME_SLICED_THREADS    : int
//        WARP_TIME_SLICED_ITEMS      : int
//        INSERT_PADDING              : bool
//        PADDING_ITEMS               : int
//    }
//
//    [<ReflectedDefinition>]
//    static member Init(block_threads:int, p.ITEMS_PER_THREAD:int, warp_time_slicing:bool) =
//        let log_warp_threads            = CUB_PTX_LOG_WARP_THREADS
//        let warp_threads                = 1 <<< LOG_WARP_THREADS
//        let warps                       = (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
//        let c.LOG_SMEM_BANKS              = CUB_PTX_c.LOG_SMEM_BANKS
//        let smem_banks                  = 1 <<< c.LOG_SMEM_BANKS
//        let tile_items                  = block_threads * p.ITEMS_PER_THREAD
//        let time_slices                 = if warp_time_slicing then warps else 1
//        let time_sliced_threads         = if warp_time_slicing then CUB_MIN block_threads WARP_THREADS else block_threads
//        let time_sliced_items           = time_sliced_threads * p.ITEMS_PER_THREAD
//        let warp_time_sliced_threads    = CUB_MIN block_threads WARP_THREADS
//        let warp_time_sliced_items      = warp_time_sliced_threads * p.ITEMS_PER_THREAD
//        let insert_padding              = ((p.ITEMS_PER_THREAD &&& (p.ITEMS_PER_THREAD - 1)) = 0)
//        let padding_items               = if insert_padding then (time_sliced_items >>> c.LOG_SMEM_BANKS) else 0
//        {   
//            LOG_WARP_THREADS            = log_warp_threads
//            WARP_THREADS                = warp_threads
//            WARPS                       = warps
//            c.LOG_SMEM_BANKS              = c.LOG_SMEM_BANKS
//            SMEM_BANKS                  = SMEM_BANKS
//            TILE_ITEMS                  = tile_items
//            TIME_SLICES                 = time_slices
//            TIME_SLICED_THREADS         = time_sliced_threads
//            TIME_SLICED_ITEMS           = time_sliced_items
//            WARP_TIME_SLICED_THREADS    = warp_time_sliced_threads
//            WARP_TIME_SLICED_ITEMS      = warp_time_sliced_items
//            INSERT_PADDING              = insert_padding
//            PADDING_ITEMS               = padding_items
//        }
//
////    static member Default(block_threads:int, p.ITEMS_PER_THREAD:int) =
////        BlockExchange.Init(block_threads, p.ITEMS_PER_THREAD, false)
////
////
//////let inlineStorage() = cuda { return! <@ fun (n:int) -> __shared__.Array(n) |> __array_to_ptr @> |> Compiler.DefineFunction }
////
//[<Record>]
//type ThreadFields =
//    {
//        mutable temp_storage    : deviceptr<int>
//        mutable linear_tid      : int
//        mutable warp_id         : int
//        mutable warp_lane       : int
//        mutable warp_offset     : int
//    }
//
//    static member Default() =
//        {
//            temp_storage    = __null()
//            linear_tid      = 0
//            warp_id         = 0
//            warp_lane       = 0
//            warp_offset     = 0
//        }

//    [<ReflectedDefinition>]
//    static member Init(c:Constants) =
//        let linear_tid = threadIdx.x
//        let warp_id = linear_tid >>> c.LOG_WARP_THREADS
//        {   temp_storage = __null()
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (c.WARP_THREADS - 1)
//            warp_id = linear_tid >>> c.LOG_WARP_THREADS
//            warp_offset = warp_id * c.WARP_TIME_SLICED_ITEMS }
//
//    [<ReflectedDefinition>]
//    static member Init(c:Constants, temp_storage:deviceptr<int>) =
//        let linear_tid = threadIdx.x
//        let warp_id = linear_tid >>> c.LOG_WARP_THREADS
//        {   temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (c.WARP_THREADS - 1)
//            warp_id = linear_tid >>> c.LOG_WARP_THREADS
//            warp_offset = warp_id * c.WARP_TIME_SLICED_ITEMS }            
//        
//    [<ReflectedDefinition>]
//    static member Init(c:Constants, linear_tid:int) =
//        let warp_id = linear_tid >>> c.LOG_WARP_THREADS
//        {   temp_storage = __null()
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (c.WARP_THREADS - 1)
//            warp_id = linear_tid >>> c.LOG_WARP_THREADS
//            warp_offset = warp_id * c.WARP_TIME_SLICED_ITEMS }
//
//    [<ReflectedDefinition>]
//    static member Init(c:Constants, temp_storage:deviceptr<int>, linear_tid:int) =
//        let warp_id = linear_tid >>> c.LOG_WARP_THREADS
//        {   temp_storage = temp_storage
//            linear_tid = linear_tid
//            warp_lane = linear_tid &&& (c.WARP_THREADS - 1)
//            warp_id = linear_tid >>> c.LOG_WARP_THREADS
//            warp_offset = warp_id * c.WARP_TIME_SLICED_ITEMS }    
//
//let PrivateStorage() = __null()
//
//[<Record>]
//type BlockExchange =
//    {
//        BLOCK_THREADS       : int
//        p.ITEMS_PER_THREAD    : int
//        WARP_TIME_SLICING   : bool
//        Constants           : Constants
//        ThreadFields        : ThreadFields
//    }
//
//    member this.Initialize() =
//        let linear_tid              = threadIdx.x
//        let warp_threads            = this.Constants.WARP_THREADS
//        let log_warp_threads        = this.Constants.LOG_WARP_THREADS
//        let warp_time_sliced_items  = this.Constants.WARP_TIME_SLICED_ITEMS
//        let warp_id                 = linear_tid >>> log_warp_threads
//        this.ThreadFields.temp_storage  <- PrivateStorage()
//        this.ThreadFields.linear_tid    <- linear_tid
//        this.ThreadFields.warp_lane     <- linear_tid &&& (warp_threads - 1)
//        this.ThreadFields.warp_id       <- linear_tid >>> log_warp_threads
//        this.ThreadFields.warp_offset   <- warp_id * warp_time_sliced_items
//        this
//
//    member this.Initialize(temp_storage:deviceptr<int>) =
//        let linear_tid              = threadIdx.x
//        let warp_threads            = this.Constants.WARP_THREADS
//        let log_warp_threads        = this.Constants.LOG_WARP_THREADS
//        let warp_time_sliced_items  = this.Constants.WARP_TIME_SLICED_ITEMS
//        let warp_id                 = linear_tid >>> log_warp_threads
//        this.ThreadFields.temp_storage  <- temp_storage
//        this.ThreadFields.linear_tid    <- linear_tid
//        this.ThreadFields.warp_lane     <- linear_tid &&& (warp_threads - 1)
//        this.ThreadFields.warp_id       <- linear_tid >>> log_warp_threads
//        this.ThreadFields.warp_offset   <- warp_id * warp_time_sliced_items
//        this
//
//    member this.Initialize(linear_tid:int) =
//        let warp_threads            = this.Constants.WARP_THREADS
//        let log_warp_threads        = this.Constants.LOG_WARP_THREADS
//        let warp_time_sliced_items  = this.Constants.WARP_TIME_SLICED_ITEMS
//        let warp_id                 = linear_tid >>> log_warp_threads
//        this.ThreadFields.temp_storage  <- PrivateStorage()
//        this.ThreadFields.linear_tid    <- linear_tid
//        this.ThreadFields.warp_lane     <- linear_tid &&& (warp_threads - 1)
//        this.ThreadFields.warp_id       <- linear_tid >>> log_warp_threads
//        this.ThreadFields.warp_offset   <- warp_id * warp_time_sliced_items
//        this
//
//    member this.Initialize(temp_storage:deviceptr<int>, linear_tid:int) =
//        let warp_threads            = this.Constants.WARP_THREADS
//        let log_warp_threads        = this.Constants.LOG_WARP_THREADS
//        let warp_time_sliced_items  = this.Constants.WARP_TIME_SLICED_ITEMS
//        let warp_id                 = linear_tid >>> log_warp_threads
//        this.ThreadFields.temp_storage  <- temp_storage
//        this.ThreadFields.linear_tid    <- linear_tid
//        this.ThreadFields.warp_lane     <- linear_tid &&& (warp_threads - 1)
//        this.ThreadFields.warp_id       <- linear_tid >>> log_warp_threads
//        this.ThreadFields.warp_offset   <- warp_id * warp_time_sliced_items
//        this
//
//
//    member this.StripedToBlocked(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let p.ITEMS_PER_THREAD    = this.p.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//        
//        stripedToBlocked
//        <|||    (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//        
//
//    member this.BlockedToStriped(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let p.ITEMS_PER_THREAD    = this.p.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        blockedToStriped
//        <|||    (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.WarpStripedToBlocked(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let p.ITEMS_PER_THREAD    = this.p.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        warpStripedToBlocked
//        <|||    (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.BlockedToWarpStriped(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let p.ITEMS_PER_THREAD    = this.p.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        blockedToWarpStriped
//        <|||    (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.ScatterToBlocked(items:deviceptr<int>, ranks:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let p.ITEMS_PER_THREAD    = this.p.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        scatterToBlocked
//        <|||    (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.ScatterToStriped(items:deviceptr<int>, ranks:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let p.ITEMS_PER_THREAD    = this.p.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        scatterToStriped
//        <|||    (block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.ScatterToStriped(items:deviceptr<int>, ranks:deviceptr<int>, is_valid:deviceptr<int>, valid_items:int) =
//        let block_threads       = this.BLOCK_THREADS
//        let p.ITEMS_PER_THREAD    = this.p.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//        let insert_padding      = this.Constants.INSERT_PADDING
//        let c.LOG_SMEM_BANKS      = this.Constants.c.LOG_SMEM_BANKS
//
//        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
//            let mutable item_offset = ranks.[ITEM]
//            if insert_padding then item_offset <- (item_offset, c.LOG_SMEM_BANKS, item_offset) |||> SHR_ADD
//            if is_valid.[ITEM] <> 0 then temp_storage.[item_offset] <- items.[ITEM]
//
//        __syncthreads()
//
//        for ITEM = 0 to p.ITEMS_PER_THREAD - 1 do
//            let mutable item_offset = (ITEM * block_threads) + linear_tid
//            if item_offset < valid_items then
//                if insert_padding then item_offset <- (item_offset, c.LOG_SMEM_BANKS, item_offset) |||> SHR_ADD
//                items.[ITEM] <- temp_storage.[item_offset]
//
//
//    static member Create(block_threads:int, p.ITEMS_PER_THREAD:int, warp_time_slicing:bool) =
//        {
//            BLOCK_THREADS       = block_threads
//            p.ITEMS_PER_THREAD    = p.ITEMS_PER_THREAD
//            WARP_TIME_SLICING   = warp_time_slicing
//            Constants           = Constants.Init(block_threads, p.ITEMS_PER_THREAD, warp_time_slicing)
//            ThreadFields        = ThreadFields.Default()
//        }
//
//
//    static member Create(block_threads:int, p.ITEMS_PER_THREAD:int) =
//        {
//            BLOCK_THREADS       = block_threads
//            p.ITEMS_PER_THREAD    = p.ITEMS_PER_THREAD
//            WARP_TIME_SLICING   = false
//            Constants           = Constants.Init(block_threads, p.ITEMS_PER_THREAD, false)
//            ThreadFields        = ThreadFields.Default()
//        }
////[<Record>]
//type BlockExchange =
//    {
//        BLOCK_THREADS : int
//        p.ITEMS_PER_THREAD : int
//        WARP_TIME_SLICING : bool
//    }
//
//    [<ReflectedDefinition>]
//    member this.BlockToStriped(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS,this.p.ITEMS_PER_THREAD,this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let linear_tid = vars.linear_tid
//        let warp_id = vars.warp_id
//        let warp_lane = vars.warp_lane
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.p.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ITEM * this.BLOCK_THREADS + linear_tid
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let temp_items = __local__.Array(this.p.ITEMS_PER_THREAD)
//                
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.p.ITEMS_PER_THREAD) + ITEM
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
//                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
//                                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        
//    [<ReflectedDefinition>]
//    member this.BlockTOWarpStriped(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.p.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let warp_lane = vars.warp_lane
//        let warp_offset = vars.warp_offset
//        let warp_id = vars.warp_id
//                        
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.p.ITEMS_PER_THREAD)
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    __syncthreads()
//                        
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * this.p.ITEMS_PER_THREAD)
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
//        
//    [<ReflectedDefinition>]
//    member this.StripedToBlocked(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.p.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let linear_tid = vars.linear_tid
//        let warp_id = vars.warp_id
//        let warp_lane = vars.warp_lane
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.p.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let props = BlockExchangec.Init(this.BLOCK_THREADS, this.p.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//                let temp_items = __local__.Array(this.p.ITEMS_PER_THREAD)
//
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
//                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
//                            
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
//                                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.p.ITEMS_PER_THREAD) + ITEM
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        
//    [<ReflectedDefinition>]
//    member this.WarpStripedToBlocked(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.p.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let warp_offset = vars.warp_offset
//        let warp_lane = vars.warp_lane
//        let warp_id = vars.warp_id
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.p.ITEMS_PER_THREAD)
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * this.p.ITEMS_PER_THREAD)
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
//        
//    [<ReflectedDefinition>]
//    member this.ScatterToBlocked(items:deviceptr<int>, ranks:deviceptr<int>, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.p.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.p.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let temp_items = __local__.Array(this.p.ITEMS_PER_THREAD)
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    let SLICE_OFFSET = c.TIME_SLICED_ITEMS * SLICE
//
//                    for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
//                            if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.p.ITEMS_PER_THREAD) + ITEM
//                            if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//
//    [<ReflectedDefinition>]
//    member this.ScatterToStriped(items:deviceptr<int>, ranks:deviceptr<int>) = //, linear_tid:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.p.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let linear_tid = vars.linear_tid
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let temp_items = __local__.Array(this.p.ITEMS_PER_THREAD)
//
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
//                            if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
//                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
//                                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.p.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]