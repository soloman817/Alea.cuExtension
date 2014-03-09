module Test.Alea.cuExtension.CUB.Block.Load
    
open Microsoft.FSharp.Quotations

open Alea.CUDA
open Alea.CUDA.Utilities

open NUnit.Framework

open Alea.cuExtension.CUB.Block

let BLOCKS = 1
let THREADS = 32
let N = BLOCKS * THREADS

let BLOCK_THREADS = THREADS
let ITEMS_PER_THREAD = 4
//type BlockLoadAlgorithm = Template.BlockLoadAlgorithm
type API<'T> = 
    {
        Default         : Function<int -> deviceptr<'T> -> deviceptr<'T> -> unit>
        Guarded         : Function<int -> deviceptr<'T> -> deviceptr<'T> -> int -> unit>
        GuardedWithOOB  : Function<int -> deviceptr<'T> -> deviceptr<'T> -> int -> 'T -> unit>
    }



let test (lp:LaunchParam) (api:Template<API<'T>>) = cuda {
    let! api = api
    let! default_kernel =
        <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) ->
            api.Default.Invoke threadIdx.x input output
        @> |> Compiler.DefineKernel        

    let! guarded_kernel =
        <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) ->
            api.Default.Invoke threadIdx.x input output
        @> |> Compiler.DefineKernel

    let! guardedwithoob_kernel =
        <@ fun (input:deviceptr<'T>) (output:deviceptr<'T>) ->
            api.Default.Invoke threadIdx.x input output
        @> |> Compiler.DefineKernel

    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let kernels = [default_kernel; guarded_kernel; guardedwithoob_kernel] |> List.map (fun k -> program.Apply k)

        fun (input:'T[]) ->
            use input = worker.Malloc(input)
            use output = worker.Malloc(input.Length)
            
            let hout =
                kernels |> List.map (fun e -> 
                    e.Launch lp input.Ptr output.Ptr
                    output.Gather())
            hout
    )}

module Sig =
    type DefaultExpr<'T> = Expr<int -> deviceptr<'T> -> deviceptr<'T> -> unit>
    type GuardedExpr<'T> = Expr<int -> deviceptr<'T> -> deviceptr<'T> -> int -> unit>
    type GuardedWithOOB<'T> = Expr<int -> deviceptr<'T> -> deviceptr<'T> -> int -> 'T -> unit>

let api (_Default:Sig.DefaultExpr<'T>) (_Guarded:Sig.GuardedExpr<'T>) (_GuardedWithOOB:Sig.GuardedWithOOB<'T>) = cuda {
    let! _Default = 
        <@ fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) ->
            (%_Default) linear_tid block_itr items
        @> |> Compiler.DefineFunction

    let! _Guarded =
        <@ fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) ->
            (%_Guarded) linear_tid block_itr items valid_items
        @> |> Compiler.DefineFunction

    let! _GuardedWithOOB =
        <@ fun (linear_tid:int) (block_itr:deviceptr<'T>) (items:deviceptr<'T>) (valid_items:int) (oob_default:'T) ->
            (%_GuardedWithOOB) linear_tid block_itr items valid_items oob_default
        @> |> Compiler.DefineFunction
            
    return { Default = _Default; Guarded = _Guarded; GuardedWithOOB = _GuardedWithOOB }
    }


[<Test>]
let ``block load template verification`` () =
    //let BlockLoad = BlockLoad.template<int>
    let template block_threads items_per_thread = cuda {
        
        //let BlockLoad = BlockLoad.API<int>.Init(block_threads, items_per_thread, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT, false).Default
        let! blockLoad = BlockLoad.template<int> block_threads items_per_thread BlockLoadAlgorithm.BLOCK_LOAD_DIRECT false

        let! kernel = 
            <@ fun (input:deviceptr<int>) (output:deviceptr<int>) ->
                blockLoad.Default.Invoke threadIdx.x output input
            @> |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            fun (input:int[]) ->
                use input = worker.Malloc(input)
                use output = worker.Malloc(input.Length)

                let lp = LaunchParam(1,1)
                
                kernel.Launch lp input.Ptr output.Ptr

                output.Gather()
        )}

    let program = template 1 8 |> Compiler.load Worker.Default
    let input = Array.init 32 (fun i -> i)
    let output = program.Run input
    
    printfn "%A" output
    


let [<Test>] ``load direct blocked`` () = 
    let _Template = Alea.cuExtension.CUB.Block.Load._Template<int>.Init(BLOCK_THREADS, ITEMS_PER_THREAD, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT,  false)
    let _Default = <@ LoadDirectBlocked.Default _Template @>
    let _Guarded = <@ LoadDirectBlocked.Guarded _Template @>
    let _GuardedWithOOB = <@ LoadDirectBlocked.GuardedWithOOB _Template @>

    let api = api _Default _Guarded _GuardedWithOOB
    let lp = LaunchParam(BLOCKS,THREADS)
    let program = (lp, api) ||> test |> Compiler.load Worker.Default
    let input = Array.init N (fun i -> i)
    let output = program.Run input

    printfn "%A" output
             
let [<Test>] ``load direct blocked vectorized`` () =
    let _Template = Alea.cuExtension.CUB.Block.Load._Template<int>.Init(BLOCK_THREADS, ITEMS_PER_THREAD, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT,  false)
    let _Default = <@ LoadDirectBlockedVectorized.Default _Template @>
    let _Guarded = <@ LoadDirectBlocked.Guarded _Template @>
    let _GuardedWithOOB = <@ LoadDirectBlocked.GuardedWithOOB _Template @>

    let api = api _Default _Guarded _GuardedWithOOB
    let lp = LaunchParam(BLOCKS,THREADS)
    let program = (lp, api) ||> test |> Compiler.load Worker.Default
    let input = Array.init N (fun i -> i)
    let output = program.Run input

    printfn "%A" output

let [<Test>] ``load direct striped`` () =
    let _Template = Alea.cuExtension.CUB.Block.Load._Template<int>.Init(BLOCK_THREADS, ITEMS_PER_THREAD, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT,  false)
    let _Default = <@ LoadDirectStriped.Default _Template @>
    let _Guarded = <@ LoadDirectStriped.Guarded _Template @>
    let _GuardedWithOOB = <@ LoadDirectStriped.GuardedWithOOB _Template @>

    let api = api _Default _Guarded _GuardedWithOOB
    let lp = LaunchParam(BLOCKS,THREADS)
    let program = (lp, api) ||> test |> Compiler.load Worker.Default
    let input = Array.init N (fun i -> i)
    let output = program.Run input

    printfn "%A" output

let [<Test>] ``load direct warp striped`` () =
    let _Template = Alea.cuExtension.CUB.Block.Load._Template<int>.Init(BLOCK_THREADS, ITEMS_PER_THREAD, BlockLoadAlgorithm.BLOCK_LOAD_DIRECT,  false)
    let _Default = <@ LoadDirectWarpStriped.Default _Template @>
    let _Guarded = <@ LoadDirectWarpStriped.Guarded _Template @>
    let _GuardedWithOOB = <@ LoadDirectWarpStriped.GuardedWithOOB _Template @>

    let api = api _Default _Guarded _GuardedWithOOB
    let lp = LaunchParam(BLOCKS,THREADS)
    let program = (lp, api) ||> test |> Compiler.load Worker.Default
    let input = Array.init N (fun i -> i)
    let output = program.Run input

    printfn "%A" output

//
//let blockLoad = Load.blockLoad
//
//
//
////type LoadDirectBlocked = 
////    | Default of int * deviceptr<int> * deviceptr<int>
////    | Guarded of int * deviceptr<int> * deviceptr<int> * int
////    | GuardedWithOOB of int * deviceptr<int> * deviceptr<int> * int * int
//
//
//
////let loadDirectBlocked = cuda {
////    let!
////    }
//
////type LoadDirectBlocked = 
////    | Default of Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>
////    | Guarded of Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>
////    | GuardedWithOOB of Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> int -> unit>
//
//
//
//
////type IBlockLoad =
////    abstract LoadDirectBlocked : LoadDirectBlocked
////
////let load (ldb:LoadDirectBlocked) (template:Template<int>) =
////    ldb |> function
////    | Default expr -> ()
////    | Guarded expr -> ()
////    | GuardedWithOOB expr -> ()
//
////let load (ldb:LoadDirectBlocked) = cuda {
////        let f = ldb |> function
////            | Default -> 
////    }
//
////let [<Test>] ``load direct blocked ``() = ()
////let [<Test>] ``load direct blocked vectorized``() = ()
////let [<Test>] ``load direct striped ``() = ()
////let [<Test>] ``load direct warp striped ``() = ()
////let loadDirectBlocked (block_threads:int) (items_per_thread:int) = 
////    fun (valid_items:int option) (oob_default:int option) ->
////        match valid_items, oob_default with
////        | None, None ->
////            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
////                //fun _ ->
////                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////            
////        | Some valid_items, None ->
////            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
////                //fun _ ->
////                let bounds = valid_items - (linear_tid * items_per_thread)
////                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////            
////        | Some valid_items, Some oob_default ->
////            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
////                //fun _ ->
////                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
////                let bounds = valid_items - (linear_tid * items_per_thread)
////                for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////            
////        | _, _ ->
////            fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) -> 
////                ()
////                //fun _ -> ()
//
////
////let loadDirectBlocked (block_threads:int) (items_per_thread:int) = 
////    fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int option) (oob_default:int option) ->
////        (valid_items, oob_default) |> function
////        | None, None ->            
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////            
////        | Some valid_items, None ->
////            let bounds = valid_items - (linear_tid * items_per_thread)
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////            
////        | Some valid_items, Some oob_default ->
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
////            let bounds = valid_items - (linear_tid * items_per_thread)
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////            
////        | _, _ ->
////            ()
//
//
////let loadInternal (algorithm:Algo) =
////    fun (block_threads:int) (items_per_thread:int) ->
////        algorithm |> function
////        | BLOCK_LOAD_DIRECT ->           (block_threads, items_per_thread) ||> loadDirectBlocked
////
////
////let blockLoad (block_threads:int) (items_per_thread:int) (algorithm:Algo) (warp_time_slicing:bool) =
////    algorithm |> function
////    | BLOCK_LOAD_DIRECT ->
////        fun (valid_items:int option) (oob_default:int option) ->
////            algorithm |> loadInternal 
////            <|| (items_per_thread, block_threads)
////            <|| (valid_items, oob_default)
////            <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
////                let foo = algorithm |> loadInternal
////                let foo = foo <||     (items_per_thread, block_threads)
////                let foo = foo <||     (valid_items, oob_default)
////                foo linear_tid block_itr items
////            @>
//
////type IBlockLoad =
////    abstract Load : int * deviceptr<int> * deviceptr<int> -> Template<Function<int -> deviceptr<int> -> deviceptr<int> -> unit>>
//////    abstract Load : int * deviceptr<int> * deviceptr<int> * int -> unit
//////    abstract Load : int * deviceptr<int> * deviceptr<int> * int * int -> unit
////
////
////let inline bload block_threads items_per_thread algorithm warp_time_slicing =
////    { new IBlockLoad with
////        member this.Load(linear_tid, block_itr, items) =
////            let load = (blockLoad block_threads items_per_thread algorithm warp_time_slicing)
////                        <|| (None, None)
////            cuda { return! load |> Compiler.DefineFunction }
////    }
//
////
////[<Record>]
////type BLoad =
////    {
////        BLOCK_THREADS       : int
////        ITEMS_PER_THREAD    : int
////        [<RecordExcludedField>] ALGORITHM           : Algo
////        WARP_TIME_SLICING   : bool
////
////        BlockLoad : int option -> int option -> int -> deviceptr<int> -> deviceptr<int> -> unit
////    }
////
////    [<ReflectedDefinition>]
////    member this.Load(linear_tid, block_itr, items) = 
////        //(blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
////        let load = this.BlockLoad
////                    <||     (None, None)
////                    <|||    (linear_tid, block_itr, items)
////        cuda { return fun _ -> load  }
////        
////    
//////    [<ReflectedDefinition>]    
//////    member this.Load(linear_tid, block_itr, items, valid_items) =
//////        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//////        <|| (Some valid_items, None)
//////        |>  Compiler.DefineFunction
//////
//////    [<ReflectedDefinition>]
//////    member this.Load(linear_tid, block_itr, items, valid_items, oob_default) =
//////        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//////        <|| (Some valid_items, Some oob_default)
//////        |>  Compiler.DefineFunction
////
////    [<ReflectedDefinition>]
////    static member Create(block_threads, items_per_thread, algorithm) =
////        {
////            BLOCK_THREADS = block_threads
////            ITEMS_PER_THREAD = items_per_thread
////            ALGORITHM   = algorithm
////            WARP_TIME_SLICING = false
////            BlockLoad = (blockLoad block_threads items_per_thread algorithm false)
////        }
//
//
////type LoadDirectBlocked =
////    {
////        Default : Function<int -> deviceptr<int> -> deviceptr<int> -> unit>
////        Guarded : Function<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>
////        GuardedWithOOB : Function<int -> deviceptr<int> -> deviceptr<int> -> int -> int -> unit>
////    }
//
////type Load =
////    | Default of Expr<int -> deviceptr<int> -> deviceptr<int> -> unit>
////    | Guarded of Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> unit>
////    | GuardedWithOOB of Expr<int -> deviceptr<int> -> deviceptr<int> -> int -> int -> unit>
//
////let loadDirectBlocked items_per_thread = cuda {
////    
////    let! dDefault = 
////        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////        @> |> Compiler.DefineFunction
////
////    let! dGuarded =
////        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
////            let bounds = valid_items - (linear_tid * items_per_thread)
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////        @> |> Compiler.DefineFunction
////        
////    let! dGuardedWithOOB =
////        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) (oob_default:int) ->
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
////            let bounds = valid_items - (linear_tid * items_per_thread)
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////        @> |> Compiler.DefineFunction
////        
////    return {Default = dDefault; Guarded = dGuarded; GuardedWithOOB = dGuardedWithOOB}}
//
////module LoadDirectBlocked =
////    
////    let Default items_per_thread = 
////        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) ->
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////        @>
////
////    let Guarded items_per_thread =
////        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) ->
////            let bounds = valid_items - (linear_tid * items_per_thread)
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////        @> |> Compiler.DefineFunction
////        
////    let GuardedWithOOB items_per_thread =
////        <@ fun (linear_tid:int) (block_itr:deviceptr<int>) (items:deviceptr<int>) (valid_items:int) (oob_default:int) ->
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- oob_default
////            let bounds = valid_items - (linear_tid * items_per_thread)
////            for ITEM = 0 to (items_per_thread - 1) do items.[ITEM] <- block_itr.[(linear_tid * items_per_thread) + ITEM]
////        @>
//        
////
////let load algorithm =
////    algorithm |> function
////    | BLOCK_LOAD_DIRECT -> LoadDirectBlocked
//
//
////let derp (load:Load) (algo:Algo) = cuda {
////    let ld =
////        load |> function
////        | Load.Default(ld) -> ld
////        | Load.Guarded(ld) -> ld
////        | Load.GuardedWithOOB(ld) -> ld
////    
////
////    }
//
//
//
//
////[<Record>]
////type BLoad =
////    {
////        BLOCK_THREADS       : int
////        ITEMS_PER_THREAD    : int
////        [<RecordExcludedField>] ALGORITHM           : Algo
////        WARP_TIME_SLICING   : bool
////        LDB                 : LoadDirectBlocked
////        //BlockLoad : int option -> int option -> int -> deviceptr<int> -> deviceptr<int> -> unit
////    }
////
////    [<ReflectedDefinition>]
////    member this.Load(linear_tid, block_itr, items) = 
////        this.LDB.Default.Invoke linear_tid block_itr items
////        
////    
//////    [<ReflectedDefinition>]    
//////    member this.Load(linear_tid, block_itr, items, valid_items) =
//////        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//////        <|| (Some valid_items, None)
//////        |>  Compiler.DefineFunction
//////
//////    [<ReflectedDefinition>]
//////    member this.Load(linear_tid, block_itr, items, valid_items, oob_default) =
//////        (blockLoad this.BLOCK_THREADS this.ITEMS_PER_THREAD this.ALGORITHM this.WARP_TIME_SLICING)
//////        <|| (Some valid_items, Some oob_default)
//////        |>  Compiler.DefineFunction
////
////    [<ReflectedDefinition>]
////    static member Create(block_threads, items_per_thread, algorithm, ldb) =
////        {
////            BLOCK_THREADS = block_threads
////            ITEMS_PER_THREAD = items_per_thread
////            ALGORITHM   = algorithm
////            WARP_TIME_SLICING = false
////            LDB                 = ldb
////        }
//
//[<Test>]
//let ``block load basic (non-record)`` () =
//
//    let template block_threads items_per_thread = cuda {
//        let! loadDirectBlocked = 
//            (   blockLoad
//                <||| (block_threads, items_per_thread, BLOCK_LOAD_DIRECT)
//                <|   false).Default
////        let! load = 
////            <@
////                blockLoad
////                <|| (block_threads, items_per_thread)
////                <|| (BLOCK_LOAD_DIRECT, false)
////            @>
////            |> Compiler.DefineFunction
////        let! load = 
////            loadDirectBlocked
////            <|| (block_threads, items_per_thread)
////            <|| (None, None)
////            |> Compiler.DefineFunction  
////        let! load = <@ fun _ -> loadDirectBlocked @> |> Compiler.DefineFunction
////        let load = (bload block_threads items_per_thread BLOCK_LOAD_DIRECT false).Load
//        let! kernel = 
//            <@ fun (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
////                let temp_storage_load = __shared__.Extern()
////                let blockLoad = blockLoad () threadIdx.x
//
//                let data = __local__.Array<int>(items_per_thread)
//                loadDirectBlocked.Invoke threadIdx.x d_in (data |> __array_to_ptr)
//                //load.Invoke threadIdx.x d_in (data |> __array_to_ptr) None None
//                //load (threadIdx.x, d_in, (data |> __array_to_ptr))
////                load(threadIdx.x, d_in, (data |> __array_to_ptr)).Invoke() 
////                blockLoad.Load(threadIdx.x, d_in, (data |> __array_to_ptr)).Invoke()
//                __syncthreads()
//
//
//            @> |> Compiler.DefineKernel
//
//        return Entry(fun program ->
//            let worker = program.Worker
//            let kernel = program.Apply kernel
//
//            let run (input:int[]) =
//                use d_in = worker.Malloc(input)
//                use d_out = worker.Malloc<int>(input.Length + 1)
////                let blockLoad = BLoad.Create(block_threads, items_per_thread, BLOCK_LOAD_DIRECT, ldb)
//                kernel.Launch (LaunchParam(2, 32)) d_in.Ptr d_out.Ptr
//
//                d_out.Gather()
//
//            run
//        )}
//
//    let BLOCK_THREADS = 32
//    let ITEMS_PER_THREAD = 1
//    let TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
//
//    let h_in = Array.init TILE_SIZE (fun i -> i)
//    let h_out = Array.scan (fun r e -> r + e) 0
//
//    printfn "Input: %A" h_in
//
//    let program = template BLOCK_THREADS ITEMS_PER_THREAD |> Compiler.load Worker.Default
//    let output = program.Run h_in
//
//    printfn "Device Output: %A" output
//
//
////[<Test>]
////let ``block load basic`` () =
////    
////    let template block_threads items_per_thread = cuda {
////        let BlockLoad = BlockLoad.Create(block_threads, items_per_thread, BLOCK_LOAD_WARP_TRANSPOSE)
////        
//// 
////
////        let! kernel = 
////            <@ fun (blockLoad:BlockLoad) (d_in:deviceptr<int>) (d_out:deviceptr<int>) ->
////                let temp_storage_load = __shared__.Extern()
////
////                let data = __local__.Array<int>(items_per_thread)
////                blockLoad.Initialize(temp_storage_load).Load(d_in, d_out)
////
////                __syncthreads()
////
////
////            @> |> Compiler.DefineKernel
////
////        return Entry(fun program ->
////            let worker = program.Worker
////            let kernel = program.Apply kernel
////
////            let run (input:int[]) =
////                use d_in = worker.Malloc(input)
////                use d_out = worker.Malloc<int>(input.Length + 1)
////
////                kernel.Launch (LaunchParam(2, 32)) BlockLoad d_in.Ptr d_out.Ptr
////
////                d_out.Gather()
////
////            run
////        )}
////
////    let BLOCK_THREADS = 32
////    let ITEMS_PER_THREAD = 1
////    let TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
////
////    let h_in = Array.init TILE_SIZE (fun i -> i)
////    let h_out = Array.scan (fun r e -> r + e) 0
////
////    printfn "Input: %A" h_in
////
////    let program = template BLOCK_THREADS ITEMS_PER_THREAD |> Compiler.load Worker.Default
////    let output = program.Run h_in
////
////    printfn "Device Output: %A" output
////
//
