module Sample.CUB.BlockRadixSort
//
//// Simple demonstration of cub::BlockSort
//// Example compilation string:
//// nvcc example_block_radix_sort.cu -gencode=arch=compute_20,code=\"sm_20,compute_20\" -o example_block_radix_sort -m32 -Xptxas -v -I../cub
//
////---------------------------------------------------------------------
//// Globals, constants and typedefs
////---------------------------------------------------------------------
//
///// Verbose output
//let g_verbose = false
//
///// Timing iterations
//let g_iterations = 100
//
///// Default grid size
//let g_grid_size = 1
//
///// Uniform key samples
//let mutable g_uniform_keys = false
//
////---------------------------------------------------------------------
//// Kernels
////---------------------------------------------------------------------
//
//(*
// * Simple kernel for performing a block-wide sorting over integers
// *)
//
//let blockSort (BLOCK_THREADS:int) (ITEMS_PER_THREAD:int) =
////__launch_bounds__ (BLOCK_THREADS)
//    let kernel =
//        <@ fun  (d_in:deviceptr<Key>)           // Tile of input
//                (d_out:deviceptr<Key>)          // Tile of output
//                (d_elapsed:deviceptr<clock_t>)  // Elapsed cycle count of block scan
//                ->
//    
//                //enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
//
//
//                // Parameterize BlockRadixSort type for our thread block
//                //typedef BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
//                //let blockRadixSort = blockRadixSort BLOCK_THREADS ITEMS_PER_THREAD
//
//                // Shared memory
//                //__shared__ typename BlockRadixSortT::TempStorage temp_storage;
//                
//                // Per-thread tile items
//                //Key items[ITEMS_PER_THREAD];
//                //let items = Array.init<Key> ITEMS_PER_THREAD (fun i -> i)
//                
//                // Our current block's offset
//                let block_offset = blockIdx.x * TILE_SIZE
//                
//                // Load items in blocked fashion
////            #if CUB_PTX_ARCH >= 350
////                LoadBlocked<LOAD_LDG>(threadIdx.x, d_in + block_offset, items);
////            #else
////                LoadBlockedVectorized<LOAD_DEFAULT>(threadIdx.x, d_in + block_offset, items);
////            #endif
//
//                // Start cycle timer
//                //clock_t start = clock();
//
//                // Sort keys
//                //BlockRadixSortT(temp_storage).SortBlockedToStriped(items);
//                
//                // Stop cycle timer
//                clock_t stop = clock();
//                
//                // Store output in striped fashion
//                //StoreStriped<STORE_DEFAULT, BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items);
//                
//                // Store elapsed clocks
//                if threadIdx.x = 0 then
//                    d_elapsed.[blockIdx.x] <- if (start > stop) then start - stop else stop - start
//        @>
//
//
////---------------------------------------------------------------------
//// Host utilities
////---------------------------------------------------------------------
//
////
//// Initialize sorting problem (and solution).
////
//let initialize
//    (h_in:Key[])
//    (h_reference:Key[])
//    (num_items:int)
//    (tile_size:int)
//    =    
//    
//    for i = 0 to (num_items - 1) do
//        if g_uniform_keys then
//            h_in.[i] <- 0
//        else
//            //RandomBits(h_in.[i])        
//        h_reference.[i] <- h_in.[i]
//    
//
//
//    // Only sort the first tile
//    //std::sort(h_reference, h_reference + tile_size)
//
//
//
//
////
//// Test BlockScan
////
//let testBlockScan (BLOCK_THREADS:int) (ITEMS_PER_THREAD:int) =
//    let test() =
//        let TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
//    
//        // Allocate host arrays
//        let h_in = Array.init<Key> (TILE_SIZE * g_grid_size) (fun _ -> new Key())
//        let h_reference = Array.init<Key> (TILE_SIZE * g_grid_size) (fun _ -> new Key())
//        let h_elapsed = Array.init<clock_t> (g_grid_size) (fun _ -> new clock_t())
//        
//        // Initialize problem and reference output on host
//        initialize h_in h_reference (TILE_SIZE * g_grid_size) TILE_SIZE
//        
//        // Initialize device arrays
//        //Key *d_in       = NULL;
//        //Key *d_out      = NULL;
//        //clock_t *d_elapsed  = NULL;
//        //CubDebugExit(cudaMalloc((void**)&d_in,          sizeof(Key) * TILE_SIZE * g_grid_size));
//        //CubDebugExit(cudaMalloc((void**)&d_out,         sizeof(Key) * TILE_SIZE * g_grid_size));
//        //CubDebugExit(cudaMalloc((void**)&d_elapsed,     sizeof(clock_t) * g_grid_size));
//
//
//        // Display input problem data
//        if g_verbose then
//            printfn "Input data: "
//            for i = 0 to (TILE_SIZE - 1) do
//                //std::cout << h_in[i] << ", ";
//            printfn "\n\n"
//        
//        // CUDA device props
//        //Device device;
//        //int max_sm_occupancy;
//        //CubDebugExit(device.Init());
//        //CubDebugExit(device.MaxSmOccupancy(max_sm_occupancy, BlockSortKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD>, BLOCK_THREADS));
//        
//        // Copy problem to device
//        //CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(Key) * TILE_SIZE * g_grid_size, cudaMemcpyHostToDevice));
//        
//        printfn "BlockRadixSort %d items (%d timing iterations, %d blocks, %d threads, %d items per thread, %d SM occupancy):\n"
//            (TILE_SIZE * g_grid_size) g_iterations g_grid_size BLOCK_THREADS ITEMS_PER_THREAD max_sm_occupancy
//        //fflush(stdout);
//        
//        // Run kernel once to prime caches and check result
////        BlockSortKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(
////            d_in,
////            d_out,
////            d_elapsed);
//
//        // Check for kernel errors and STDIO from the kernel, if any
//        //CubDebugExit(cudaDeviceSynchronize());
//
//        // Check results
//        printfn "\tOutput items: "
//        //let compare = CompareDeviceResults(h_reference, d_out, TILE_SIZE, g_verbose, g_verbose);
//        printfn "%s\n" (if compare then "FAIL" else "PASS")
//        //AssertEquals(0, compare);
//        //fflush(stdout);
//        
//        // Run this several times and average the performance results
//        //GpuTimer            timer;
//        //float               elapsed_millis          = 0.0;
//        //unsigned long long  elapsed_clocks          = 0;
//
////        for i = 0 to (g_iterations - 1) do
////            timer.Start();
//            
//            // Run kernel
////            BlockSortKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD><<<g_grid_size, BLOCK_THREADS>>>(
////                d_in,
////                d_out,
////                d_elapsed);
//
////            timer.Stop();
////            elapsed_millis += timer.ElapsedMillis();
//
//
//            // Copy clocks from device
////            CubDebugExit(cudaMemcpy(h_elapsed, d_elapsed, sizeof(clock_t) * g_grid_size, cudaMemcpyDeviceToHost));
////            for (int i = 0; i < g_grid_size; i++)
////                elapsed_clocks += h_elapsed[i];
////        }
//
//
//        // Check for kernel errors and STDIO from the kernel, if any
////        CubDebugExit(cudaDeviceSynchronize());
//
//
//        // Display timing results
//        let avg_millis            = elapsed_millis / g_iterations
//        let avg_items_per_sec     = float(TILE_SIZE * g_grid_size) / avg_millis / 1000.0
//        let avg_clocks           = double(elapsed_clocks) / g_iterations / g_grid_size
//        let avg_clocks_per_item  = avg_clocks / TILE_SIZE
//
//
//        printfn "\tAverage BlockRadixSort::SortBlocked clocks: %.3f\n" avg_clocks
//        printfn "\tAverage BlockRadixSort::SortBlocked clocks per item: %.3f\n" avg_clocks_per_item
//        printfn "\tAverage kernel millis: %.4f\n" avg_millis
//        printfn "\tAverage million items / sec: %.4f\n" avg_items_per_sec
//        //fflush(stdout);
//
//
//        // Cleanup
////        if (h_in) delete[] h_in;
////        if (h_reference) delete[] h_reference;
////        if (h_elapsed) delete[] h_elapsed;
////        if (d_in) CubDebugExit(cudaFree(d_in));
////        if (d_out) CubDebugExit(cudaFree(d_out));
////        if (d_elapsed) CubDebugExit(cudaFree(d_elapsed));
