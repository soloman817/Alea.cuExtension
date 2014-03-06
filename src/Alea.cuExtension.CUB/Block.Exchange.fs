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
        type API<'T> =
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
                API<'T>.Init<'T>(block_threads, items_per_thread, false)
    
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
            static member Init(tp:Params.API<'T>) =
                let log_warp_threads            = CUB_PTX_LOG_WARP_THREADS
                let warp_threads                = 1 <<< log_warp_threads
                let warps                       = (tp.BLOCK_THREADS + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
                let log_smem_banks              = CUB_PTX_LOG_SMEM_BANKS
                let smem_banks                  = 1 <<< log_smem_banks
                let tile_items                  = tp.BLOCK_THREADS * tp.ITEMS_PER_THREAD
                let time_slices                 = if tp.WARP_TIME_SLICING then warps else 1
                let time_sliced_threads         = if tp.WARP_TIME_SLICING then (tp.BLOCK_THREADS, warp_threads) ||> CUB_MIN else tp.BLOCK_THREADS
                let time_sliced_items           = time_sliced_threads * tp.ITEMS_PER_THREAD
                let warp_time_sliced_threads    = (tp.BLOCK_THREADS, warp_threads) ||> CUB_MIN
                let warp_time_sliced_items      = warp_time_sliced_threads * tp.ITEMS_PER_THREAD
                let insert_padding              = ((tp.ITEMS_PER_THREAD &&& (tp.ITEMS_PER_THREAD - 1)) = 0)
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
            static member Uninitialized(tp:Params.API<'T>) =
                let c = Constants.API.Init tp
                let length = c.TIME_SLICED_ITEMS + c.PADDING_ITEMS
                let s = __shared__.Array(length)
                let ptr = s |> __array_to_ptr
                { Ptr = ptr; Length = length}

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
            static member Init<'T>(temp_storage:TempStorage.API<'T>, linear_tid, warp_lane, warp_id, warp_offset) =
                {
                    temp_storage    = temp_storage
                    linear_tid      = linear_tid
                    warp_lane       = warp_lane
                    warp_id         = warp_id
                    warp_offset     = warp_offset
                }

            [<ReflectedDefinition>]
            static member Init(tp:Params.API<'T>) = API<'T>.Init<'T>(TempStorage.API<'T>.Uninitialized<'T>(tp), 0, 0, 0, 0)

    type _TemplateParams<'T>    = Params.API<'T>
    type _Constants             = Constants.API
    type _TempStorage<'T>       = TempStorage.API<'T>
    type _ThreadFields<'T>      = ThreadFields.API<'T>

module private Internal =
    module Sig =
        module BlockToStriped =
            type Default<'T>            = deviceptr<'T> -> unit
            type WithTimeslicing<'T>    = deviceptr<'T> -> unit

        module BlockToWarpStriped =
            type Default<'T>            = BlockToStriped.Default<'T>
            type WithTimeslicing<'T>    = BlockToStriped.WithTimeslicing<'T>

        module StripedToBlocked =
            type Default<'T>            = BlockToStriped.Default<'T>
            type WithTimeslicing<'T>    = BlockToStriped.WithTimeslicing<'T>

        module WarpStripedToBlocked =
            type Default<'T>            = BlockToStriped.Default<'T>
            type WithTimeslicing<'T>    = BlockToStriped.WithTimeslicing<'T>

        module ScatterToBlocked =
            type Default<'T>            = deviceptr<'T> -> deviceptr<int> -> unit
            type WithTimeslicing<'T>    = deviceptr<'T> -> deviceptr<int> -> unit

        module ScatterToStriped =
            type Default<'T>            = ScatterToBlocked.Default<'T>
            type WithTimeslicing<'T>    = ScatterToBlocked.WithTimeslicing<'T>


module BlockedToStriped =
    open Template
    open Internal

    type API<'T> =
        {
            Default             : Sig.BlockToStriped.Default<'T>
            WithTimeslicing     : Sig.BlockToStriped.WithTimeslicing<'T>
        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) =
        let c = _Constants.Init tp
        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (tf.linear_tid * tp.ITEMS_PER_THREAD) + ITEM
            if c.INSERT_PADDING then 
                item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            tf.temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = ITEM * tp.BLOCK_THREADS + tf.linear_tid
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- tf.temp_storage.[item_offset]

    let [<ReflectedDefinition>] inline WithTimeslicing (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) =
        let c = _Constants.Init tp
        let temp_items = __local__.Array(tp.ITEMS_PER_THREAD)
                
        for SLICE = 0 to (c.TIME_SLICES - 1) do
            let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
            let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS

            __syncthreads()

            if tf.warp_id = SLICE then
                for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (tf.warp_lane * tp.ITEMS_PER_THREAD) + ITEM
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    tf.temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                let STRIP_OFFSET = ITEM * tp.BLOCK_THREADS
                let STRIP_OOB = STRIP_OFFSET + tp.BLOCK_THREADS

                if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                    let mutable item_offset = STRIP_OFFSET + tf.linear_tid - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        temp_items.[ITEM] <- tf.temp_storage.[item_offset]

        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            items.[ITEM] <- temp_items.[ITEM]


    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                Default         =   Default tp tf
                WithTimeslicing =   WithTimeslicing tp tf
            }


module BlockedToWarpStriped =
    open Template
    open Internal

    type API<'T> =
        {
            Default             : Sig.BlockToWarpStriped.Default<'T>
            WithTimeslicing     : Sig.BlockToWarpStriped.WithTimeslicing<'T>
        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) =
        let c = _Constants.Init tp
        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = tf.warp_offset + ITEM + (tf.warp_lane * tp.ITEMS_PER_THREAD)
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- tf.temp_storage.[item_offset]

        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = tf.warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + tf.warp_lane
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- tf.temp_storage.[item_offset]

    let [<ReflectedDefinition>] inline WithTimeslicing (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) =
        let c = _Constants.Init tp
        for SLICE = 0 to (c.TIME_SLICES - 1) do
            __syncthreads()
                        
            if tf.warp_id = SLICE then
                for ITEM = 0 to (tp.ITEMS_PER_THREAD- 1) do
                    let mutable item_offset = ITEM + (tf.warp_lane * tp.ITEMS_PER_THREAD)
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    tf.temp_storage.[item_offset] <- items.[ITEM]

                for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + tf.warp_lane
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    items.[ITEM] <- tf.temp_storage.[item_offset]


    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
        {
            Default         =   Default tp tf
            WithTimeslicing =   WithTimeslicing tp tf
        }


module StripedToBlocked =
    open Template
    open Internal

    type API<'T> =
        {
            Default             : Sig.StripedToBlocked.Default<'T>
            WithTimeslicing     : Sig.StripedToBlocked.WithTimeslicing<'T>
        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) =
        let c = _Constants.Init tp
        for ITEM = 0 to (tp.ITEMS_PER_THREAD- 1) do
            let mutable item_offset = (ITEM * tp.BLOCK_THREADS) + tf.linear_tid
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            tf.temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (tf.linear_tid * tp.ITEMS_PER_THREAD) + ITEM
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- tf.temp_storage.[item_offset]
            

    let [<ReflectedDefinition>] inline WithTimeslicing (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) =
        let c = _Constants.Init tp
        let temp_items = __local__.Array<'T>(tp.ITEMS_PER_THREAD)

        for SLICE = 0 to (c.TIME_SLICES - 1) do
            let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
            let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS

            __syncthreads()

            for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                let STRIP_OFFSET = ITEM * tp.BLOCK_THREADS
                let STRIP_OOB = STRIP_OFFSET + tp.BLOCK_THREADS
                            
                if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                    let mutable item_offset = STRIP_OFFSET + tf.linear_tid - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
                        if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                        tf.temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            if tf.warp_id = SLICE then
                for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = (tf.warp_lane * tp.ITEMS_PER_THREAD) + ITEM
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    temp_items.[ITEM] <- tf.temp_storage.[item_offset]

        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            items.[ITEM] <- temp_items.[ITEM]


    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
        {
            Default         =   Default tp tf
            WithTimeslicing =   WithTimeslicing tp tf
        }


module WarpStripedToBlocked =
    open Template
    open Internal

    type API<'T> =
        {
            Default             : Sig.WarpStripedToBlocked.Default<'T>
            WithTimeslicing     : Sig.WarpStripedToBlocked.WithTimeslicing<'T>
        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) =
        let c = _Constants.Init tp
        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = tf.warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + tf.warp_lane
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            tf.temp_storage.[item_offset] <- items.[ITEM]

        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = tf.warp_offset + ITEM + (tf.warp_lane * tp.ITEMS_PER_THREAD)
            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
            items.[ITEM] <- tf.temp_storage.[item_offset]

    let [<ReflectedDefinition>] inline WithTimeslicing (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) =
        let c = _Constants.Init tp
        for SLICE = 0 to (c.TIME_SLICES - 1) do
            __syncthreads()

            if tf.warp_id = SLICE then
                for ITEM = 0 to (tp.ITEMS_PER_THREAD- 1) do
                    let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + tf.warp_lane
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    tf.temp_storage.[item_offset] <- items.[ITEM]

                for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ITEM + (tf.warp_lane * tp.ITEMS_PER_THREAD)
                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                    items.[ITEM] <- tf.temp_storage.[item_offset]


    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
        {
            Default         =   Default tp tf
            WithTimeslicing =   WithTimeslicing tp tf
        }


module ScatterToBlocked =
    open Template
    open Internal

    type API<'T> =
        {
            Default             : Sig.ScatterToBlocked.Default<'T>
            WithTimeslicing     : Sig.ScatterToBlocked.WithTimeslicing<'T>
        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let c = _Constants.Init tp
        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = ranks.[ITEM]
            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            tf.temp_storage.[item_offset] <- items.[ITEM]

        __syncthreads()

        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            let mutable item_offset = (tf.linear_tid * tp.ITEMS_PER_THREAD) + ITEM
            if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
            items.[ITEM] <- tf.temp_storage.[item_offset]


    let [<ReflectedDefinition>] inline WithTimeslicing (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) (ranks:deviceptr<int>) =
        let c = _Constants.Init tp
        let temp_items = __local__.Array<'T>(tp.ITEMS_PER_THREAD)
        for SLICE = 0 to (c.TIME_SLICES - 1) do
            __syncthreads()

            let SLICE_OFFSET = c.TIME_SLICED_ITEMS * SLICE

            for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
                    if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                    tf.temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()


            if tf.warp_id = SLICE then
                for ITEM = 0 to (tp.ITEMS_PER_THREAD- 1) do
                    let mutable item_offset = (tf.warp_lane * tp.ITEMS_PER_THREAD) + ITEM
                    if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                    temp_items.[ITEM] <- tf.temp_storage.[item_offset]

        for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
            items.[ITEM] <- temp_items.[ITEM]


    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>) 
        (tf:_ThreadFields<'T>) =
        {
            Default         =   Default tp tf
            WithTimeslicing =   WithTimeslicing tp tf
        }


module ScatterToStriped =
    open Template
    open Internal

    type API<'T> =
        {
            Default             : Sig.ScatterToStriped.Default<'T>
            WithTimeslicing     : Sig.ScatterToStriped.WithTimeslicing<'T>
        }

    let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) (ranks:deviceptr<int>) =
            let c = _Constants.Init tp
            for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = ranks.[ITEM]
                if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                tf.temp_storage.[item_offset] <- items.[ITEM]

            __syncthreads()

            for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                let mutable item_offset = (ITEM * tp.BLOCK_THREADS) + tf.linear_tid
                if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                items.[ITEM] <- tf.temp_storage.[item_offset]


    let private WithTimeslicing (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (items:deviceptr<'T>) (ranks:deviceptr<int>) =
            let c = _Constants.Init tp
            let temp_items = __local__.Array(tp.ITEMS_PER_THREAD)
            for SLICE = 0 to (c.TIME_SLICES - 1) do
                let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
                let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS

                __syncthreads()

                for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                    let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
                    if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
                        if c.INSERT_PADDING then item_offset <- __ptx__.SHR_ADD( (item_offset |> uint32), c.LOG_SMEM_BANKS |> uint32, (item_offset |> uint32)) |> int
                        tf.temp_storage.[item_offset] <- items.[ITEM]

                __syncthreads()

                for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                    let STRIP_OFFSET = ITEM * tp.BLOCK_THREADS
                    let STRIP_OOB = STRIP_OFFSET + tp.BLOCK_THREADS

                    if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
                        let mutable item_offset = STRIP_OFFSET + tf.linear_tid - SLICE_OFFSET
                        if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
                            temp_items.[ITEM] <- tf.temp_storage.[item_offset]

            for ITEM = 0 to (tp.ITEMS_PER_THREAD - 1) do
                items.[ITEM] <- temp_items.[ITEM]


    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                Default         =   Default tp tf
                WithTimeslicing =   WithTimeslicing tp tf
            }


module BlockExchange =
    open Template

    [<Record>]
    type API<'T> =
        {
            BlockedToStriped        : BlockedToStriped.API<'T>
            BlockedToWarpStriped    : BlockedToWarpStriped.API<'T>
            StripedToBlocked        : StripedToBlocked.API<'T>
            WarpStripedToBlocked    : WarpStripedToBlocked.API<'T>
            ScatterToBlocked        : ScatterToBlocked.API<'T>
            ScatterToStriped        : ScatterToStriped.API<'T>
        }

//        [<ReflectedDefinition>]
//        static member Create(block_threads, items_per_thread, warp_time_slicing) =
//            let tp = _TemplateParams<'T>.Init(block_threads, items_per_thread, warp_time_slicing)
//            let tf = _ThreadFields<'T>.Init(tp)
//            {
//                BlockedToStriped        =   BlockedToStriped.api tp tf
//                BlockedToWarpStriped    =   BlockedToWarpStriped.api tp tf
//                StripedToBlocked        =   StripedToBlocked.api tp tf
//                WarpStripedToBlocked    =   WarpStripedToBlocked.api tp tf                                                            
//                ScatterToBlocked        =   ScatterToBlocked.api tp tf
//                ScatterToStriped        =   ScatterToStriped.api tp tf
//            }

    let [<ReflectedDefinition>] api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                BlockedToStriped        =   BlockedToStriped.api tp tf
                BlockedToWarpStriped    =   BlockedToWarpStriped.api tp tf
                StripedToBlocked        =   StripedToBlocked.api tp tf
                WarpStripedToBlocked    =   WarpStripedToBlocked.api tp tf                                                            
                ScatterToBlocked        =   ScatterToBlocked.api tp tf
                ScatterToStriped        =   ScatterToStriped.api tp tf
            }

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
//    fun block_threads items_per_thread -> 
//        fun (linear_tid:int option) ->
//            match linear_tid with
//            | Some linear_tid ->
//                (linear_tid |> Some |> warp_id) * ((block_threads, items_per_thread, ()) |||> WARP_TIME_SLICED_ITEMS)
//            | None ->
//                (threadIdx.x |> Some |> warp_id) * ((block_threads, items_per_thread, ()) |||> WARP_TIME_SLICED_ITEMS)



//let blockedToStriped (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING      = ((),items_per_thread,()) |||> INSERT_PADDING
//    let TIME_SLICED_ITEMS   = (block_threads, items_per_thread, warp_time_slicing) |||> TIME_SLICED_ITEMS
//    let TIME_SLICES         = (block_threads, (), warp_time_slicing) |||> TIME_SLICES
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (linear_tid * items_per_thread) + ITEM
//                    if INSERT_PADDING then 
//                        item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = ITEM * block_threads + linear_tid
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let temp_items = __local__.Array(items_per_thread)
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
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (warp_lane * items_per_thread) + ITEM
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let STRIP_OFFSET = ITEM * block_threads
//                        let STRIP_OOB = STRIP_OFFSET + block_threads
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
//                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
        


//let blockedToWarpStriped (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING              = ((), items_per_thread, ())                |||> INSERT_PADDING
//    let WARP_TIME_SLICED_THREADS    = (block_threads, (), ())                   |||> WARP_TIME_SLICED_THREADS
//    let TIME_SLICES                 = (block_threads, (), warp_time_slicing)    |||> TIME_SLICES
//
//    let warp_offset = (block_threads, items_per_thread) ||> warp_offset
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_offset = linear_tid |> Some |> warp_offset
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * items_per_thread)
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
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
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * items_per_thread)
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
        
//
//let stripedToBlocked (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING      = ((), items_per_thread, ())                              |||> INSERT_PADDING
//    let TIME_SLICES         = (block_threads, (), warp_time_slicing)                  |||> TIME_SLICES
//    let TIME_SLICED_ITEMS   = (block_threads, items_per_thread, warp_time_slicing)    |||> TIME_SLICED_ITEMS
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (ITEM * block_threads) + linear_tid
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (linear_tid * items_per_thread) + ITEM
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let temp_items = __local__.Array(items_per_thread)
//
//            fun (items:deviceptr<int>) ->
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let STRIP_OFFSET = ITEM * block_threads
//                        let STRIP_OOB = STRIP_OFFSET + block_threads
//                            
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
//                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                                temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    let warp_id = linear_tid |> Some |> warp_id
//                    let warp_lane = linear_tid |> Some |> warp_lane
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (warp_lane * items_per_thread) + ITEM
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        

//let warpStripedToBlocked (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let WARP_TIME_SLICED_THREADS    = (block_threads, (), ())                   |||> WARP_TIME_SLICED_THREADS
//    let INSERT_PADDING              = ((), items_per_thread, ())                |||> INSERT_PADDING
//    let TIME_SLICES                 = (block_threads, (), warp_time_slicing)    |||> TIME_SLICES
//    let warp_offset                 = (block_threads, items_per_thread)          ||> warp_offset
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) ->
//                let warp_offset = linear_tid |> Some |> warp_offset
//                let warp_lane = linear_tid |> Some |> warp_lane
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * items_per_thread)
//                    if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
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
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + warp_lane
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * items_per_thread)
//                            if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
        
//
//let scatterToBlocked (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING          = ((), items_per_thread, ())                            |||> INSERT_PADDING
//    let TIME_SLICES             = (block_threads, (), warp_time_slicing)                |||> TIME_SLICES
//    let TIME_SLICED_ITEMS       = (block_threads, items_per_thread, warp_time_slicing)  |||> TIME_SLICED_ITEMS
//    let WARP_TIME_SLICED_ITEMS  = (block_threads, items_per_thread, ())                 |||> WARP_TIME_SLICED_ITEMS
//
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (linear_tid * items_per_thread) + ITEM
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) ->
//                let temp_items = __local__.Array(items_per_thread)
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    let SLICE_OFFSET = TIME_SLICED_ITEMS * SLICE
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS) then
//                            if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    let warp_id = linear_tid |> Some |> warp_id
//                    let warp_lane = linear_tid |> Some |> warp_lane
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (items_per_thread - 1) do
//                            let mutable item_offset = (warp_lane * items_per_thread) + ITEM
//                            if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]


//let scatterToStriped (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
//    let INSERT_PADDING          = ((), block_threads, ())                               |||> INSERT_PADDING
//    let TIME_SLICES             = (block_threads, (), warp_time_slicing)                |||> TIME_SLICES
//    let TIME_SLICED_ITEMS       = (block_threads, items_per_thread, warp_time_slicing)  |||> TIME_SLICED_ITEMS
//    let WARP_TIME_SLICED_ITEMS  = (block_threads, items_per_thread, ())                 |||> WARP_TIME_SLICED_ITEMS
//    
//    match warp_time_slicing with
//    | false ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) (is_valid:deviceptr<int>) (valid_items:int) ->    
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    let mutable item_offset = (ITEM * block_threads) + linear_tid
//                    if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//    | true ->
//        fun (temp_storage:deviceptr<int>) (linear_tid:int) ->
//            let temp_items = __local__.Array(items_per_thread)
//            fun (items:deviceptr<int>) (ranks:deviceptr<int>) (is_valid:deviceptr<int>) (valid_items:int) ->    
//                for SLICE = 0 to (TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS) then
//                            if INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (items_per_thread - 1) do
//                        let STRIP_OFFSET = ITEM * block_threads
//                        let STRIP_OOB = STRIP_OFFSET + block_threads
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS) then
//                                if INSERT_PADDING then item_offset <- item_offset + (item_offset >>> LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (items_per_thread - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]

//let vars (temp_storage:deviceptr<int> option) (linear_tid:int option) =
//    match temp_storage, linear_tid with
//    | Some temp_storage, Some linear_tid -> temp_storage,       linear_tid
//    | None,              Some linear_tid -> privateStorage(),   linear_tid
//    | Some temp_storage, None ->            temp_storage,       threadIdx.x
//    | None,              None ->            privateStorage(),   threadIdx.x

//let blockExchange (block_threads:int) (items_per_thread:int) (warp_time_slicing:bool) =
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
//        LOG_SMEM_BANKS              : int
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
//    static member Init(block_threads:int, items_per_thread:int, warp_time_slicing:bool) =
//        let log_warp_threads            = CUB_PTX_LOG_WARP_THREADS
//        let warp_threads                = 1 <<< LOG_WARP_THREADS
//        let warps                       = (block_threads + CUB_PTX_WARP_THREADS - 1) / CUB_PTX_WARP_THREADS
//        let log_smem_banks              = CUB_PTX_LOG_SMEM_BANKS
//        let smem_banks                  = 1 <<< LOG_SMEM_BANKS
//        let tile_items                  = block_threads * items_per_thread
//        let time_slices                 = if warp_time_slicing then warps else 1
//        let time_sliced_threads         = if warp_time_slicing then CUB_MIN block_threads WARP_THREADS else block_threads
//        let time_sliced_items           = time_sliced_threads * items_per_thread
//        let warp_time_sliced_threads    = CUB_MIN block_threads WARP_THREADS
//        let warp_time_sliced_items      = warp_time_sliced_threads * items_per_thread
//        let insert_padding              = ((items_per_thread &&& (items_per_thread - 1)) = 0)
//        let padding_items               = if insert_padding then (time_sliced_items >>> LOG_SMEM_BANKS) else 0
//        {   
//            LOG_WARP_THREADS            = log_warp_threads
//            WARP_THREADS                = warp_threads
//            WARPS                       = warps
//            LOG_SMEM_BANKS              = log_smem_banks
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
////    static member Default(block_threads:int, items_per_thread:int) =
////        BlockExchange.Init(block_threads, items_per_thread, false)
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
//        ITEMS_PER_THREAD    : int
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
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//        
//        stripedToBlocked
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//        
//
//    member this.BlockedToStriped(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        blockedToStriped
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.WarpStripedToBlocked(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        warpStripedToBlocked
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.BlockedToWarpStriped(items:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        blockedToWarpStriped
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.ScatterToBlocked(items:deviceptr<int>, ranks:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        scatterToBlocked
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.ScatterToStriped(items:deviceptr<int>, ranks:deviceptr<int>) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//
//        scatterToStriped
//        <|||    (block_threads, items_per_thread, warp_time_slicing)
//        <||     (temp_storage, linear_tid)
//
//
//    member this.ScatterToStriped(items:deviceptr<int>, ranks:deviceptr<int>, is_valid:deviceptr<int>, valid_items:int) =
//        let block_threads       = this.BLOCK_THREADS
//        let items_per_thread    = this.ITEMS_PER_THREAD
//        let warp_time_slicing   = this.WARP_TIME_SLICING
//        let temp_storage        = this.ThreadFields.temp_storage
//        let linear_tid          = this.ThreadFields.linear_tid
//        let insert_padding      = this.Constants.INSERT_PADDING
//        let log_smem_banks      = this.Constants.LOG_SMEM_BANKS
//
//        for ITEM = 0 to items_per_thread - 1 do
//            let mutable item_offset = ranks.[ITEM]
//            if insert_padding then item_offset <- (item_offset, log_smem_banks, item_offset) |||> SHR_ADD
//            if is_valid.[ITEM] <> 0 then temp_storage.[item_offset] <- items.[ITEM]
//
//        __syncthreads()
//
//        for ITEM = 0 to items_per_thread - 1 do
//            let mutable item_offset = (ITEM * block_threads) + linear_tid
//            if item_offset < valid_items then
//                if insert_padding then item_offset <- (item_offset, log_smem_banks, item_offset) |||> SHR_ADD
//                items.[ITEM] <- temp_storage.[item_offset]
//
//
//    static member Create(block_threads:int, items_per_thread:int, warp_time_slicing:bool) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            WARP_TIME_SLICING   = warp_time_slicing
//            Constants           = Constants.Init(block_threads, items_per_thread, warp_time_slicing)
//            ThreadFields        = ThreadFields.Default()
//        }
//
//
//    static member Create(block_threads:int, items_per_thread:int) =
//        {
//            BLOCK_THREADS       = block_threads
//            ITEMS_PER_THREAD    = items_per_thread
//            WARP_TIME_SLICING   = false
//            Constants           = Constants.Init(block_threads, items_per_thread, false)
//            ThreadFields        = ThreadFields.Default()
//        }
////[<Record>]
//type BlockExchange =
//    {
//        BLOCK_THREADS : int
//        ITEMS_PER_THREAD : int
//        WARP_TIME_SLICING : bool
//    }
//
//    [<ReflectedDefinition>]
//    member this.BlockToStriped(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS,this.ITEMS_PER_THREAD,this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let linear_tid = vars.linear_tid
//        let warp_id = vars.warp_id
//        let warp_lane = vars.warp_lane
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ITEM * this.BLOCK_THREADS + linear_tid
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let temp_items = __local__.Array(this.ITEMS_PER_THREAD)
//                
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
//                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
//                                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        
//    [<ReflectedDefinition>]
//    member this.BlockTOWarpStriped(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let warp_lane = vars.warp_lane
//        let warp_offset = vars.warp_offset
//        let warp_id = vars.warp_id
//                        
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
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
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
//        
//    [<ReflectedDefinition>]
//    member this.StripedToBlocked(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let linear_tid = vars.linear_tid
//        let warp_id = vars.warp_id
//        let warp_lane = vars.warp_lane
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//                let temp_items = __local__.Array(this.ITEMS_PER_THREAD)
//
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
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
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//        
//    [<ReflectedDefinition>]
//    member this.WarpStripedToBlocked(items:deviceptr<int>) = //, linear_tid:int, warp_id:int, warp_lane:int, warp_offset:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let warp_offset = vars.warp_offset
//        let warp_lane = vars.warp_lane
//        let warp_id = vars.warp_id
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = warp_offset + ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                    if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                    items.[ITEM] <- temp_storage.[item_offset]
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (ITEM * c.WARP_TIME_SLICED_THREADS) + warp_lane
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = ITEM + (warp_lane * this.ITEMS_PER_THREAD)
//                            if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                            items.[ITEM] <- temp_storage.[item_offset]
//        
//    [<ReflectedDefinition>]
//    member this.ScatterToBlocked(items:deviceptr<int>, ranks:deviceptr<int>, linear_tid:int, warp_id:int, warp_lane:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (linear_tid * this.ITEMS_PER_THREAD) + ITEM
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let temp_items = __local__.Array(this.ITEMS_PER_THREAD)
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    __syncthreads()
//
//                    let SLICE_OFFSET = c.TIME_SLICED_ITEMS * SLICE
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
//                            if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    if warp_id = SLICE then
//                        for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                            let mutable item_offset = (warp_lane * this.ITEMS_PER_THREAD) + ITEM
//                            if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]
//
//    [<ReflectedDefinition>]
//    member this.ScatterToStriped(items:deviceptr<int>, ranks:deviceptr<int>) = //, linear_tid:int) =
//        let props = BlockExchangec.Init(this.BLOCK_THREADS, this.ITEMS_PER_THREAD, this.WARP_TIME_SLICING)
//        let vars = BlockExchangeVars.Init(props)
//        let linear_tid = vars.linear_tid
//
//        match this.WARP_TIME_SLICING with
//        | false ->
//            fun (temp_storage:deviceptr<int>) ->
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = ranks.[ITEM]
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    temp_storage.[item_offset] <- items.[ITEM]
//
//                __syncthreads()
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    let mutable item_offset = (ITEM * this.BLOCK_THREADS) + linear_tid
//                    if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                    items.[ITEM] <- temp_storage.[item_offset]
//
//        | true ->
//            fun (temp_storage:deviceptr<int>) ->
//                let temp_items = __local__.Array(this.ITEMS_PER_THREAD)
//
//                for SLICE = 0 to (c.TIME_SLICES - 1) do
//                    let SLICE_OFFSET = SLICE * c.TIME_SLICED_ITEMS
//                    let SLICE_OOB = SLICE_OFFSET + c.TIME_SLICED_ITEMS
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let mutable item_offset = ranks.[ITEM] - SLICE_OFFSET
//                        if (item_offset >= 0) && (item_offset < c.WARP_TIME_SLICED_ITEMS) then
//                            if c.INSERT_PADDING then item_offset <- SHR_ADD (item_offset |> uint32) c.LOG_SMEM_BANKS (item_offset |> uint32) |> int
//                            temp_storage.[item_offset] <- items.[ITEM]
//
//                    __syncthreads()
//
//                    for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                        let STRIP_OFFSET = ITEM * this.BLOCK_THREADS
//                        let STRIP_OOB = STRIP_OFFSET + this.BLOCK_THREADS
//
//                        if (SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET) then
//                            let mutable item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET
//                            if (item_offset >= 0) && (item_offset < c.TIME_SLICED_ITEMS) then
//                                if c.INSERT_PADDING then item_offset <- item_offset + (item_offset >>> c.LOG_SMEM_BANKS)
//                                temp_items.[ITEM] <- temp_storage.[item_offset]
//
//                for ITEM = 0 to (this.ITEMS_PER_THREAD - 1) do
//                    items.[ITEM] <- temp_items.[ITEM]