[<AutoOpen>]
module Alea.cuExtension.CUB.Warp.WarpSpecializations.WarpScanSmem

open System
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities

open Alea.cuExtension.CUB.Common
open Alea.cuExtension.CUB.Utilities
open Alea.cuExtension.CUB.Thread


let inline private Broadcast (temp_storage:deviceptr<'T>) (warp_id:int) (lane_id:int) 
    (input:'T) (src_lane:int) =
    let threadStore = STORE_VOLATILE |> ThreadStore
    let threadLoad  = LOAD_DEFAULT |> ThreadLoad
    let lane_id = lane_id |> uint32
    let src_lane = src_lane |> uint32
    if lane_id = src_lane then (temp_storage + warp_id, input) ||> threadStore
    (temp_storage + warp_id) |> threadLoad
    


module Template =
    [<AutoOpen>]
    module Params =
        [<Record>]
        type API<'T> =
            {
                LOGICAL_WARPS           : int
                LOGICAL_WARP_THREADS    : int
                scan_op                 : IScanOp<'T> //('T -> 'T -> 'T)
            }

            [<ReflectedDefinition>]
            member this.Get() = (this.LOGICAL_WARPS, this.LOGICAL_WARP_THREADS, this.scan_op)

            [<ReflectedDefinition>]
            static member inline Init(logical_warps, logical_warp_threads, scan_op) =
                {
                    LOGICAL_WARPS           = logical_warps
                    LOGICAL_WARP_THREADS    = logical_warp_threads
                    scan_op                 = scan_op
                }


    [<AutoOpen>]
    module TempStorage =
        [<Record>]
        type API<'T> =
            {
                mutable Ptr                 : deviceptr<'T>
                mutable LOGICAL_WARPS       : int
                mutable WARP_SMEM_ELEMENTS  : int
            }

            member this.Item
                with    [<ReflectedDefinition>] get (idx:int) = this.Ptr.[idx]
                and     [<ReflectedDefinition>] set (idx:int) (v:'T) = this.Ptr.[idx] <- v

            member this.Get
                with    [<ReflectedDefinition>] get (i:int, j:int) = this.Ptr.[j + i * this.WARP_SMEM_ELEMENTS]
                and     [<ReflectedDefinition>] set (i:int, j:int) (v:'T) = this.Ptr.[j + i * this.WARP_SMEM_ELEMENTS] <- v
        
            [<ReflectedDefinition>]
            static member inline Init<'T>(logical_warps, warp_smem_elements) =
                let s = __shared__.Array<'T>(logical_warps*warp_smem_elements)
                let ptr = s |> __array_to_ptr
                {
                    Ptr                 = ptr
                    LOGICAL_WARPS       = logical_warps
                    WARP_SMEM_ELEMENTS  = warp_smem_elements
                }

            [<ReflectedDefinition>]
            static member inline Uninitialized<'T>() =
                {
                    Ptr                 = __null<'T>()
                    LOGICAL_WARPS       = 0
                    WARP_SMEM_ELEMENTS  = 0
                }

        type TempStorage<'T> = API<'T>


    
    [<AutoOpen>]
    module ThreadFields =
        [<Record>]
        type API<'T> =
            {
                mutable temp_storage    : TempStorage.API<'T>
                mutable warp_id         : int
                mutable lane_id         : int
            }

            [<ReflectedDefinition>]
            member this.Get() = (this.temp_storage, this.warp_id, this.lane_id)

            [<ReflectedDefinition>] 
            static member inline Init<'T>(temp_storage:TempStorage.API<'T>, warp_id, lane_id) =
                {
                    temp_storage    = temp_storage
                    warp_id         = warp_id
                    lane_id         = lane_id
                }

            [<ReflectedDefinition>] 
            static member inline Init<'T>(logical_warps, warp_smem_elements, warp_id, lane_id) =
                API<'T>.Init(TempStorage.Init(logical_warps, warp_smem_elements), warp_id, lane_id)
            
            [<ReflectedDefinition>] 
            static member inline Init<'T>(logical_warps, warp_smem_elements) =
                API<'T>.Init(TempStorage.Init(logical_warps, warp_smem_elements), 0, 0)
                

            [<ReflectedDefinition>]
            static member inline Uninitialized<'T>() =
                {
                    temp_storage    = TempStorage<'T>.Uninitialized<'T>()
                    warp_id         = 0
                    lane_id         = 0
                }

    type _TemplateParams<'T>    = Params.API<'T>
    type _TempStorage<'T>       = TempStorage.API<'T>
    type _ThreadFields<'T>      = ThreadFields.API<'T>


module private Internal =
    open Template

    module Constants =
        
        let POW_OF_TWO =
            fun logical_warp_threads -> ((logical_warp_threads &&& (logical_warp_threads - 1)) = 0)

        /// The number of warp scan steps
        let STEPS =
            fun logical_warp_threads ->
                logical_warp_threads |> log2

        /// The number of threads in half a warp
        let HALF_WARP_THREADS =
            fun logical_warp_threads ->
                let STEPS = logical_warp_threads |> STEPS
                1 <<< (STEPS - 1)

        /// The number of shared memory elements per warp
        let WARP_SMEM_ELEMENTS =
            fun logical_warp_threads ->
                logical_warp_threads + (logical_warp_threads |> HALF_WARP_THREADS)

        
    module Sig =
        module InclusiveSum =
            type Default<'T>        = 'T -> Ref<'T> -> unit
            type WithAggregate<'T>  = 'T -> Ref<'T> -> Ref<'T> -> unit
                
        module InclusiveScan =
            type Default<'T>        = InclusiveSum.Default<'T>
            type WithAggregate<'T>  = InclusiveSum.WithAggregate<'T>

        module ExclusiveScan =
            type Default<'T>        = 'T -> Ref<'T> -> 'T -> unit
            type WithAggregate<'T>  = 'T -> Ref<'T> -> 'T -> Ref<'T> -> unit

            module Identityless =
                type Default<'T>        = 'T -> Ref<'T> -> unit
                type WithAggregate<'T>  = 'T -> Ref<'T> -> Ref<'T> -> unit
                
        type Broadcast<'T> = 'T -> int -> 'T

    module InitIdentity =
    
        let [<ReflectedDefinition>] inline private True<'T> (tf:_ThreadFields<'T>) =
            let threadStore = STORE_VOLATILE |> ThreadStore
            let identity = ZeroInitialize()
            ()
            
        let [<ReflectedDefinition>] inline private False (tf:_ThreadFields<'T>) =
            ()

        let api has_identity = if has_identity then True else False


    module ScanStep =
        type private ScanStepAttribute(modifier:string) =
            inherit Attribute()

            interface ICustomCallBuilder with
                member this.Build(ctx, irObject, info, irParams) =
                    match irObject, irParams with
                    | None, irMax :: irPtr :: irVals :: [] ->
                        let max = irMax.HasObject |> function
                            | true -> irMax.Object :?> int
                            | false -> failwith "max must be constant"

                        // if we do the loop here, it is unrolled by compiler, not kernel runtime
                        // think of this job as the C++ template expanding job, same thing!
                        for i = 0 to max - 1 do
                            let irIndex = IRCommonInstructionBuilder.Instance.BuildConstant(ctx, i)
                            let irVal = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irPtr, irIndex :: [])
                            let irPtr = ()
                    
                            let irPtr = IRCommonInstructionBuilder.Instance.BuildGEP(ctx, irVals, irIndex :: [])
                            IRCommonInstructionBuilder.Instance.BuildStore(ctx, irPtr, irVal) |> ignore

                        IRCommonInstructionBuilder.Instance.BuildNop(ctx) |> Some

                    | _ -> None



    let [<ReflectedDefinition>] inline BasicScan has_identity share_final (scan_op:IScanOp<'T>) (partial:'T) =
        let scan_op = scan_op.op
        __null() |> __ptr_to_obj 
        


module InclusiveSum =
    open Template
    open Internal

    [<Record>]
    type API<'T> =
        {
            Default         : Sig.InclusiveSum.Default<'T>
            WithAggregate   : Sig.InclusiveSum.WithAggregate<'T>
        }


    let [<ReflectedDefinition>] inline private Default<'T> (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) 
        (input:'T) (output:Ref<'T>) =
            let has_identity = true //PRIMITIVE()
            let initIdentity = 
                InitIdentity.api 
                <|| (has_identity, tf)
            let basicScan = (has_identity, false, tp.scan_op) |||> BasicScan
            
            initIdentity
            output := (input, tp.scan_op) ||> basicScan
    
    let [<ReflectedDefinition>] inline private WithAggregate<'T> (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
        
        let warp_smem_elements = tp.LOGICAL_WARP_THREADS |> Constants.WARP_SMEM_ELEMENTS
        let has_identity = true //PRIMITIVE()
        let initIdentity = 
            InitIdentity.api 
            <|  has_identity
            <|  tf

        let basicScan = (has_identity, true, tp.scan_op) |||> BasicScan
        let threadLoad = LOAD_VOLATILE |> ThreadLoad
    
        //(%initIdentity)()
        output := input |> basicScan
        //@TODO
//            let w_a : 'T = tf.temp_storage.[(warp_smem_elements - 1) + warp_id * logical_warps]
        warp_aggregate := tf.temp_storage.[(warp_smem_elements - 1) + tf.warp_id * tp.LOGICAL_WARPS] //|> __obj_reinterpret |> %threadLoad //(w_a |> __unbox) // |> %threadLoad  
    

    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                Default         =   Default tp tf
                WithAggregate   =   WithAggregate tp tf
            }
            

module InclusiveScan =
    open Template
    open Internal

    type API<'T> =
        {
            Default         : Sig.InclusiveScan.Default<'T>
            WithAggregate   : Sig.InclusiveScan.WithAggregate<'T>
        }    
    
    let [<ReflectedDefinition>] inline private Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) =
        ()

    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>) =
        ()

    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                Default         =   Default tp tf
                WithAggregate   =   WithAggregate tp tf
            }


module ExclusiveScan =
    open Template
    open Internal
    
    type API<'T> =
        {
            Default             : Sig.ExclusiveScan.Default<'T>
            Default_NoID        : Sig.ExclusiveScan.Identityless.Default<'T>
            WithAggregate       : Sig.ExclusiveScan.WithAggregate<'T>
            WithAggregate_NoID  : Sig.ExclusiveScan.Identityless.WithAggregate<'T>
        }

    let [<ReflectedDefinition>] inline private Default (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (identity:'T) =
        ()

    let [<ReflectedDefinition>] inline private WithAggregate (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>)
        (input:'T) (output:Ref<'T>) (identity:'T) (warp_aggregate:Ref<'T>) =
        ()

    module private Identityless =
        let [<ReflectedDefinition>] inline Default (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>)
            (input:'T) (output:Ref<'T>) =
            ()

        let [<ReflectedDefinition>] inline WithAggregate (tp:_TemplateParams<'T>)
            (tf:_ThreadFields<'T>)
            (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<'T>)=
            ()

    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            {
                Default             =   Default tp tf
                Default_NoID        =   Identityless.Default tp tf
                WithAggregate       =   WithAggregate tp tf
                WithAggregate_NoID  =    Identityless.WithAggregate tp tf
            }


module WarpScanSmem =
    open Template
    open Internal

    [<Record>]
    type Constants =
        {
            POW_OF_TWO          : bool
            STEPS               : int
            HALF_WARP_THREADS   : int
            WARP_SMEM_ELEMENTS  : int
        }

        [<ReflectedDefinition>]
        static member Init(logical_warp_threads) =
            {
                POW_OF_TWO          = logical_warp_threads |> Constants.POW_OF_TWO
                STEPS               = logical_warp_threads |> Constants.STEPS
                HALF_WARP_THREADS   = logical_warp_threads |> Constants.HALF_WARP_THREADS
                WARP_SMEM_ELEMENTS  = logical_warp_threads |> Constants.WARP_SMEM_ELEMENTS
            }

    type API<'T> =
        {
            Constants       : Constants            
            InclusiveScan   : InclusiveScan.API<'T>
            InclusiveSum    : InclusiveSum.API<'T>
            ExclusiveScan   : ExclusiveScan.API<'T>
            Broadcast       : Internal.Sig.Broadcast<'T>
        }

    
    let [<ReflectedDefinition>] inline api (tp:_TemplateParams<'T>)
        (tf:_ThreadFields<'T>) =
            let c = Constants.Init tp.LOGICAL_WARP_THREADS
            {
                Constants       =   Constants.Init tp.LOGICAL_WARP_THREADS
                InclusiveScan   =   InclusiveScan.api tp tf
                InclusiveSum    =   InclusiveSum.api tp tf
                ExclusiveScan   =   ExclusiveScan.api tp tf
                Broadcast       =   Broadcast tf.temp_storage.Ptr tf.warp_id tf.lane_id
            }





//let inline _TempStorage() =
//    fun logical_warps warp_smem_elements ->
//        __shared__.Array2D(logical_warps, warp_smem_elements)
//
//
//let initIdentity logical_warps logical_warp_threads =
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    fun (has_identity:bool) ->
//        let temp_storage = (logical_warps, WARP_SMEM_ELEMENTS) ||> _TempStorage()
//        let store = STORE_VOLATILE |> threadStore()
//        fun warp_id lane_id ->
//            match has_identity with
//            | true ->
//                let identity = ZeroInitialize() |> __ptr_to_obj
//                (temp_storage.[warp_id, lane_id] |> __array_to_ptr, identity) ||> store
//            | false ->
//                ()
//
//let scanStep logical_warps logical_warp_threads =
//    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
//    let STEPS = logical_warp_threads |> STEPS
//    fun (temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//        let store = STORE_VOLATILE |> threadStore()
//        fun has_identity step ->
//            let step = ref 0
//            fun (partial:Ref<int>) (scan_op:(int -> int -> int)) ->
//                while !step < STEPS do
//                    let OFFSET = 1 <<< !step
//                    
//                    //(temp_storage |> __array_to_ptr, !partial) ||> store
//
//                    if has_identity || (lane_id >= OFFSET) then
//                        let addend = (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id)] |> __obj_to_ptr, Some(partial |> __ref_to_ptr)) ||> load
//                        partial := (addend.Value, !partial) ||> scan_op
//                        
//                    step := !step + 1
//
//
//let broadcast =
//    fun (temp_storage:deviceptr<int>) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//        let store = STORE_VOLATILE |> threadStore()
//        fun input src_lane ->
//            if lane_id = src_lane then (temp_storage.[warp_id] |> __obj_to_ptr, input) ||> store
//            (temp_storage.[warp_id] |> __obj_to_ptr, None) ||> load
//            |> Option.get
//
//
//let inline basicScan logical_warps logical_warp_threads = 
//    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
//    let scanStep = (logical_warps, logical_warp_threads) ||> scanStep
//    fun has_identity share_final ->
//        fun (temp_storage:int[,]) warp_id lane_id ->
//            let store = STORE_VOLATILE |> threadStore()
//            fun (partial:int) (scan_op:(int -> int -> int)) ->
//                let partial = partial |> __obj_to_ref
//                scanStep
//                <|||    (temp_storage, warp_id, lane_id)
//                <||     (has_identity, 0)
//                <||     (partial, scan_op)
//                if share_final then (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id)] |> __obj_to_ptr, !partial) ||> store
//                !partial
//
//
//let inline inclusiveSum logical_warps logical_warp_threads =
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    let initIdentity = (logical_warps, logical_warp_threads) ||> initIdentity
//    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan
//
//    fun (temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//            
//        fun (input:'T) (output:Ref<'T>) (warp_aggregate:Ref<int> option) ->
//            match warp_aggregate with
//            | None ->
//                let HAS_IDENTITY = true // Traits<int>::PRIMITIVE
//                initIdentity
//                <|  HAS_IDENTITY 
//                <|| (warp_id, lane_id)
//                
//                output :=
//                    basicScan
//                    <||     (HAS_IDENTITY, false)
//                    <|||    (temp_storage, warp_id, lane_id) 
//                    <||     (input, ( + ))
//
//            | Some warp_aggregate ->
//                let HAS_IDENTITY = true // Traits<int>::PRIMITIVE
//                initIdentity
//                <|  HAS_IDENTITY
//                <|| (warp_id, lane_id)
//
//                output :=
//                    basicScan
//                    <||     (HAS_IDENTITY, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, ( + ))
//
//                warp_aggregate :=
//                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get
//
//
//let inline inclusiveScan logical_warps logical_warp_threads =
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan
//
//    fun (temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//
//        fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (warp_aggregate:Ref<int> option) ->
//            match warp_aggregate with
//            | None ->
//                output :=
//                    basicScan
//                    <||     (false, false)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//            | Some warp_aggregate ->
//                output :=
//                    basicScan
//                    <||     (false, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                warp_aggregate :=
//                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get 
//
//    
//let inline exclusiveScan logical_warps logical_warp_threads =
//    let HALF_WARP_THREADS = logical_warp_threads |> HALF_WARP_THREADS
//    let WARP_SMEM_ELEMENTS = logical_warp_threads |> WARP_SMEM_ELEMENTS
//    let basicScan = (logical_warps, logical_warp_threads) ||> basicScan
//
//    fun (temp_storage:int[,]) warp_id lane_id ->
//        let load = LOAD_VOLATILE |> threadLoad()
//        let store = STORE_VOLATILE |> threadStore()
//
//        fun (input:'T) (output:Ref<'T>) (scan_op:(int -> int -> int)) (identity:int option) (warp_aggregate:Ref<int> option) ->
//            match identity, warp_aggregate with
//            | Some identity, None ->
//                (temp_storage.[warp_id, lane_id] |> __obj_to_ptr, identity) ||> store
//                let inclusive =
//                    basicScan
//                    <||     (true, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get
//
//            | Some identity, Some warp_aggregate ->
//                (temp_storage.[warp_id, lane_id] |> __obj_to_ptr, identity) ||> store
//                let inclusive =
//                    basicScan
//                    <||     (true, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)] |> __obj_to_ptr, None) 
//                    ||> load
//                    |> Option.get
//
//                warp_aggregate :=
//                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//            | None, None ->
//                let inclusive =
//                    basicScan
//                    <||     (false, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//            | None, Some warp_aggregate ->
//                let inclusive =
//                    basicScan
//                    <||     (false, true)
//                    <|||    (temp_storage, warp_id, lane_id)
//                    <||     (input, scan_op)
//
//                output :=
//                    (temp_storage.[warp_id, (HALF_WARP_THREADS + lane_id - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//                warp_aggregate :=
//                    (temp_storage.[warp_id, (WARP_SMEM_ELEMENTS - 1)] |> __obj_to_ptr, None)
//                    ||> load
//                    |> Option.get
//
//
//type Constants =
//    {
//        STEPS : int
//        HALF_WARP_THREADS : int
//        WARP_SMEM_ELEMENTS : int
//    }
//
//    static member Init(logical_warp_threads:int) =
//        {
//            STEPS               = logical_warp_threads |> STEPS
//            HALF_WARP_THREADS   = logical_warp_threads |> HALF_WARP_THREADS
//            WARP_SMEM_ELEMENTS  = logical_warp_threads |> WARP_SMEM_ELEMENTS
//        }
//                    
//type TempStorage = int[,]
//
//[<Record>]
//type internal ThreadFields =
//    {
//        temp_storage : TempStorage
//        warp_id : uint32
//        lane_id : uint32
//    }
//
//    static member Init(temp_storage:int[,], warp_id:uint32, lane_id:uint32) =
//        {
//            temp_storage = temp_storage
//            warp_id = warp_id
//            lane_id = lane_id
//        }
//            
//[<Record>]
//type WarpScanSmem =
//    {
//        // template parameters
//        LOGICAL_WARPS : int
//        LOGICAL_WARP_THREADS : int
//        // constants / enum
//        Constants : Constants
//        //TempStorage : int[,]
//        //ThreadFields : ThreadFields<int>
//
//    }
//
//        
//    member this.ScanStep(partial:Ref<int>) =
//        fun has_identity step -> ()
//
//    member this.ScanStep(partial:Ref<int>, scan_op:(int -> int -> int), step:bool) = 
//        fun has_identity step -> ()
//
//    member this.Broadcast(input:int, src_lane:uint32) = ()
//        
//    member this.BasicScan(partial:int, scan_op:(int -> int -> int)) = ()
//
//    member this.InclusiveSum(input:int, output:Ref<int>) = ()
//    member this.InclusiveSum(input:int, output:Ref<int>, warp_aggregate:Ref<int>) = ()
//
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) = ()
//    member this.InclusiveScan(input:int, output:Ref<int>, scan_op:(int -> int -> int)) = ()
//
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int)) = ()
//    member this.ExclusiveScan(input:int, output:Ref<int>, identity:int, scan_op:(int -> int -> int), warp_aggregate:Ref<int>) = ()
//
//
//    static member Create(logical_warps, logical_warp_threads) =
//        let c = logical_warp_threads |> Constants.Init
//        //let temp_storage = Array2D.zeroCreate logical_warps c.WARP_SMEM_ELEMENTS
//        let temp_storage = __shared__.Array2D(logical_warps, c.WARP_SMEM_ELEMENTS)
//        {
//            LOGICAL_WARPS           = logical_warps
//            LOGICAL_WARP_THREADS    = logical_warp_threads
//            Constants               = c
//            //TempStorage             = temp_storage
//            //ThreadFields            = (temp_storage, 0u, 0u) |> ThreadFields.Init
//        }