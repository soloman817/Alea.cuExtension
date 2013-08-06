module Test.Alea.CUDA.Extension.MGPU.BenchmarkStats
open Alea.CUDA.Extension.Output.Util
// Sample stats from running various algorithm benchmarks.  Used for comparison

type TestParams =
    {
        SourceCounts : int list
        Iterations : int list
        Terms : int list
    }

type BenchmarkType =
    | BulkInsert
    | BulkRemove
    | IntervalMove
//    | Join
//    | LaunchBox
    | LoadBalance
    | LocalitySort
    | Merge
    | Scan
//    | Search
//    | SegSort
//    | Sets
//    | Sort
    | SortedSearch

let bulkInsert =   { SourceCounts   = [  10000; 50000; 100000; 200000; 500000; 1000000; 2000000; 5000000; 10000000; 20000000]
                     Iterations     = [   2000;  2000;   2000;   1000;    500;     400;     400;     400;      300;      300]
                     Terms          = [      0] }

let bulkRemove =   { SourceCounts   = [  10000; 50000; 100000; 200000; 500000; 1000000; 2000000; 5000000; 10000000; 20000000]
                     Iterations     = [   2000;  2000;   2000;   1000;    500;     400;     400;     400;      300;      300]
                     Terms          = [      0] }

let intervalMove = { SourceCounts   = [  10000;   50000;  100000; 200000; 500000; 1000000; 2000000; 5000000; 10000000; 20000000]
                     Iterations     = [  10000;    1000;    5000;   5000;   5000;     200;     200;     200;     1000;     1000]
                     Terms          = [5000000; 2000000; 1000000; 500000; 200000;  100000;   50000;   20000;    10000;     5000; 2000] }

let loadBalance =  { SourceCounts   = [  10000; 50000; 100000; 200000; 500000; 1000000; 2000000; 5000000; 10000000; 20000000; 50000000]
                     Iterations     = [  50000; 40000;  20000;  10000;  10000;    5000;    5000;    4000;     3000;     3000;     2000]
                     Terms          = [      0] }

let merge =        { SourceCounts   = [  10000; 50000; 100000; 200000; 500000; 1000000; 2000000; 5000000; 10000000; 20000000]
                     Iterations     = [   1000;  1000;   1000;    500;    200;     200;     200;     200;      100;      100]
                     Terms          = [      0] }

let scan =         { SourceCounts   = [  10000; 50000; 100000; 200000; 500000; 1000000; 2000000; 5000000; 10000000; 20000000]
                     Iterations     = [   1000;  1000;   1000;    500;    200;     200;     200;     200;      100;      100]
                     Terms          = [      0] }

let sortedSearch = { SourceCounts   = [  10000; 50000; 100000; 200000; 500000; 1000000; 2000000; 5000000; 10000000; 20000000]
                     Iterations     = [  10000; 10000;  10000;   5000;   5000;    3000;    3000;    3000;     2000;     2000]
                     Terms          = [      0] }






let testParameters (bt:BenchmarkType) =
    match bt with
    | BulkInsert -> bulkInsert
    | BulkRemove -> bulkRemove
    | IntervalMove -> intervalMove
//    | Join -> join
//    | LaunchBox -> launchBox
    | LoadBalance  -> loadBalance
//    | LocalitySort -> localitySort
    | Merge -> merge
    | Scan -> scan
//    | SegSort
//    | Sets
//    | Sort -> sort
    | SortedSearch -> sortedSearch



//Tesla K20c :  705.500 Mhz   (Ordinal 0)
//13 SMs enabled. Compute Capability sm_35
//FreeMem:   3576MB   TotalMem:   3584MB.
//Mem Clock: 2600.000 Mhz x 320 bits   (208.000);)
//ECC Enabled
module TeslaK20c =
    let deviceFolderName = "teslaK20c"

                              // throughput (M/s)     bandwidth (GB/s)
    module ModernGPU =
        module ScanStats =
            let notes = ""
            let tp = testParameters(Scan)
            let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations
            let int32_stats = [(  433.734,     5.205 );     // 10k
                                   (  702.083,     8.425 );     // 50k
                                   ( 1898.394,    22.781 );     // 100k
                                   ( 4527.857,    54.334 );     // 200k
                                   ( 7650.295,    91.804 );     // 500k
                                   ( 8946.772,   107.361 );     // 1M
                                   (10262.650,   123.152 );     // 2M
                                   (10547.469,   126.570 );     // 5M
                                   (10657.556,   127.891 );     // 10M
                                   (10627.220,   127.527 ) ] 

            let int64_stats = [ (   258.734,    6.210);      
                                      (   650.588,   15.614);     
                                      (  1206.063,   28.946);     
                                      (  2560.823,   61.460);     
                                      (  3972.426,   95.338);    
                                      (  4378.827,  105.092);    
                                      (  4725.657,  113.416);    
                                      (  4877.600,  117.062);    
                                      (  4920.150,  118.084);    
                                      (  4967.462,  119.219) ]   



        module BulkRemoveStats =
            let notes = ""
            let tp = testParameters(BulkRemove)
            let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations
            let int32_stats = [  (  439.238,      3.514);
                                                   ( 1435.891,     11.487);
                                                   ( 2951.152,     23.609);
                                                   ( 4280.485,     34.244);
                                                   ( 7083.000,     56.664);
                                                   ( 9031.785,     72.254);
                                                   (10341.514,     82.732);
                                                   (11340.596,     90.725);
                                                   (11663.864,     93.311);
                                                   (11631.951,     93.056) ]

            let int64_stats = [( 430.590,      6.028);
                                                   (1594.400,     22.322);
                                                   (1897.559,     26.566);
                                                   (3873.250,     54.225);
                                                   (6107.108,     85.500);
                                                   (7563.070,    105.883);
                                                   (8548.353,    119.677);
                                                   (9257.074,    129.599);
                                                   (9484.010,    132.776);
                                                   (9470.483,    132.587) ]

            let float32_stats = [(   439.108,      3.513);
                                                     (  1627.180,     13.017);
                                                     (  2864.078,     22.913);
                                                     (  4244.398,     33.955);
                                                     (  6727.525,     53.820);
                                                     (  8936.857,     71.495);
                                                     ( 10374.727,     82.998);
                                                     ( 11345.599,     90.765);
                                                     ( 11666.451,     93.332);
                                                     ( 11630.902,     93.047) ]

            let float64_stats = [(   426.651,     5.973);
                                                     (  1631.012,    22.834);
                                                     (  2539.655,    35.555);
                                                     (  3813.503,    53.389);
                                                     (  5849.930,    81.899);
                                                     (  7454.849,   104.368);
                                                     (  8521.374,   119.299);
                                                     (  9231.647,   129.243);
                                                     (  9471.084,   132.595);
                                                     (  9448.916,   132.285) ]

            let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]
    
        module BulkInsertStats =
            let notes = ""
            let tp = testParameters(BulkInsert)
            let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations
            let int32_stats = [ (  486.572,      4.866);
                                                  ( 1659.098,     16.591);
                                                  ( 2772.436,     27.724);
                                                  ( 4029.393,     40.294);
                                                  ( 6707.667,     67.077);
                                                  (  8509.619,    85.096);
                                                  (  9593.272,    95.933);
                                                  ( 10323.136,    103.231);
                                                  ( 10300.645,    103.006);
                                                  ( 10188.207,    101.882) ]

            let int64_stats = [(  480.793,      8.654);
                                                   ( 1574.917,     28.349);
                                                   ( 2314.222,     41.656);
                                                   ( 3427.396,     61.693);
                                                   ( 5199.131,     93.584);
                                                   ( 6185.029,    111.331);
                                                   ( 6751.097,    121.520);
                                                   ( 7088.240,    127.588);
                                                   ( 7065.361,    127.176);
                                                   ( 7030.112,    126.542) ]

            let float32_stats = [(   473.354,      4.734);
                                                     (  1638.850,     16.388);
                                                     (  2775.101,     27.751);
                                                     (  4040.216,     40.402);
                                                     (  6463.839,     64.638);
                                                     (  8505.831,     85.058);
                                                     (  9592.521,     95.925);
                                                     ( 10327.269,    103.273);
                                                     ( 10303.121,    103.031);
                                                     ( 10197.278,    101.973) ]

            let float64_stats = [(   479.031,      8.623);
                                                     (  1633.738,     29.407);
                                                     (  2324.661,     41.844);
                                                     (  3526.716,     63.481);
                                                     (  5078.619,     91.415);
                                                     (  6197.664,    111.558);
                                                     (  6756.559,    121.618);
                                                     (  7098.190,    127.767);
                                                     (  7072.367,    127.303);
                                                     (  7028.623,    126.515) ]

            let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]


        module IntervalExpandStats =
            let tp = testParameters(IntervalMove)            
            module AvgSegLength25 =
                let notes = "Average segment length of 25 elements."
                let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations

                let int32_stats = [ ]

                let int64_stats = [ ]

                let float32_stats = [ ]

                let float64_stats = [ ]

                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]

            module ConstCountChangingExpandRate =
                let notes = "Constant 10M count and changing expand rate."
                let termList = tp.Terms
                let sourceCounts = Array.init termList.Length (fun _ -> 10000000)
                let nIterations = Array.init termList.Length (fun _ -> 300)

                let int32_stats = [ ]

                let int64_stats = [ ]

                let float32_stats = [ ]

                let float64_stats = [ ]

                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]

        module IntervalMoveStats =
            let tp = testParameters(IntervalMove)            
            module AvgSegLength25 =
                let notes = "Average segment length of 25 elements."
                let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations

                let int32_stats = [ ]

                let int64_stats = [ ]

                let float32_stats = [ ]

                let float64_stats = [ ]

                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]

            module ConstCountChangingExpandRate =
                let notes = "Constant 10M count and changing expand rate."
                let termList = tp.Terms
                let sourceCounts = Array.init termList.Length (fun _ -> 10000000)
                let nIterations = Array.init termList.Length (fun _ -> 300)

                let int32_stats = [ ]

                let int64_stats = [ ]

                let float32_stats = [ ]

                let float64_stats = [ ]

                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]

    module Thrust =
        module ScanStats =
            let notes = ""
            let int32_stats =  [ (    14.487,    0.174 );
                                  (   101.726,    1.221 );
                                  (   768.751,    9.225 );
                                  (  1480.935,   17.771 );
                                  (  2858.013,   34.296 );
                                  (  4125.783,   49.509 );
                                  (  4756.823,   57.082 );
                                  (  5575.663,   66.908 );
                                  (  5937.756,   71.253 );
                                  (  5985.076,   71.821 ) ]
    
            let int64_stats =    [ (    64.414,    1.546);
                                      (   309.692,    7.433);
                                      (   520.199,   12.485);
                                      (   959.225,   23.021);
                                      (  2211.916,   53.086);
                                      (  2870.851,   68.900);
                                      (  3642.104,   87.411);
                                      (  4151.910,   99.646);
                                      (  4431.106,  106.347);
                                      (  4540.052,  108.961) ]


//GeForce GTX 560 Ti : 1700.000 Mhz   (Ordinal 0)
//8 SMs enabled. Compute Capability sm_21
//FreeMem:    760MB   TotalMem:   1024MB.
//Mem Clock: 2004.000 Mhz x 256 bits   (128.256);)
//ECC Disabled
module GF560Ti = 
    let deviceFolderName = "gtx560Ti"
    
    module ModernGPU =
        let opponentName = "MGPU"

        module ScanStats =
            let notes = ""
            let tp = testParameters(Scan)
            let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations

            let int32_stats = [ ( 522.294,     6.268);    
                                    (1409.874,    16.918);     
                                    (2793.189,    33.518);     
                                    (5016.043,    60.193);     
                                    (7157.139,    85.886);     
                                    (7842.271,    94.107);     
                                    (8115.966,    97.392);     
                                    (8301.897,    99.623);     
                                    (8325.225,    99.903);     
                                    (8427.884,   101.135) ]

            let int64_stats = [ ( 364.793,     8.755);
                                      (1494.888,    35.877);  
                                      (2588.645,    62.127);  
                                      (3098.852,    74.372);  
                                      (3877.090,    93.050);  
                                      (4041.277,    96.991);  
                                      (4131.842,    99.164);  
                                      (4193.033,   100.633);  
                                      (4218.722,   101.249);  
                                      (4229.460,   101.507) ]

        module BulkRemoveStats =
            let notes = ""
            let tp = testParameters(BulkRemove)
            let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations
            let int32_stats = [    (  802.967,      6.424);
                                             ( 2479.822,     19.839);
                                             ( 3336.508,     26.692);
                                             ( 5050.460,     40.404);
                                             ( 7398.474,     59.188);
                                             ( 8682.389,     69.459);
                                             ( 9482.862,     75.863);
                                             ( 9928.677,     79.429);
                                             ( 9991.879,     79.935);
                                             (10080.749,     80.646) ]

            let int64_stats = [   ( 800.971,     11.214);
                                              (1965.858,     27.522);
                                              (2891.853,     40.486);
                                              (3855.462,     53.976);
                                              (5075.141,     71.052);
                                              (5686.893,     79.616);
                                              (6067.874,     84.950);
                                              (6239.254,     87.350);
                                              (6291.009,     88.074);
                                              (6317.008,     88.438) ]

            let float32_stats = [(  813.859,      6.511);
                                             ( 2514.249,     20.114);
                                             ( 3351.780,     26.814);
                                             ( 5065.922,     40.527);
                                             ( 7404.456,     59.236);
                                             ( 8653.136,     69.225);
                                             ( 9517.225,     76.138);
                                             ( 9927.696,     79.422);
                                             (10003.004,     80.024);
                                             (10071.503,     80.572) ]

            let float64_stats = [ ( 788.942,     11.045);
                                              (1998.952,     27.985);
                                              (2895.004,     40.530);
                                              (3872.307,     54.212);
                                              (5076.655,     71.073);
                                              (5689.991,     79.660);
                                              (6067.964,     84.951);
                                              (6255.847,     87.582);
                                              (6253.138,     87.544);
                                              (6318.662,     88.461) ]

            let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]

        module BulkInsertStats =
            let notes = ""
            let tp = testParameters(BulkInsert)
            let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations

            let int32_stats = [( 856.817,      8.568);
                                         (2043.384,     20.434);
                                         (2954.352,     29.544);
                                         (4200.184,     42.002);
                                         (5770.926,     57.709);
                                         (6639.800,     66.398);
                                         (6982.460,     69.825);
                                         (7014.502,     70.145);
                                         (7013.587,     70.136);
                                         (6994.164,     69.942) ]

            let int64_stats = [  ( 824.623,     14.843);
                                             (1762.430,     31.724);
                                             (2489.903,     44.818);
                                             (3321.832,     59.793);
                                             (4261.111,     76.700);
                                             (4684.220,     84.316);
                                             (4833.514,     87.003);
                                             (4854.146,     87.375);
                                             (4846.890,     87.244);
                                             (4826.222,     86.872) ]

            let float32_stats = [( 879.891,      8.799);
                                             (2041.054,     20.411);
                                             (2968.250,     29.682);
                                             (4189.688,     41.897);
                                             (5872.360,     58.724);
                                             (6665.170,     66.652);
                                             (6983.851,     69.839);
                                             (7001.270,     70.013);
                                             (7028.527,     70.285);
                                             (7007.654,     70.077) ]

            let float64_stats = [( 821.568,     14.788);
                                             (1759.917,     31.679);
                                             (2472.175,     44.499);
                                             (3324.487,     59.841);
                                             (4270.517,     76.869);
                                             (4686.849,     84.363);
                                             (4830.979,     86.958);
                                             (4846.682,     87.240);
                                             (4838.171,     87.087);
                                             (4834.255,     87.017) ]

            let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]

        module SortedSearchStats =
            let tp = testParameters(SortedSearch)
            let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations
            module SS1 = 
                let notes = "Aka sorted search 1" // fix later
                let int32_stats = [   (  709.298,      3.546);
                                                ( 1640.029,      8.200);
                                                ( 2527.613,     12.638);
                                                ( 3597.330,     17.987);
                                                ( 5227.086,     26.135);
                                                ( 6117.027,     30.585);
                                                ( 6170.863,     30.854);
                                                ( 6110.644,     30.553);
                                                ( 6132.644,     30.663);
                                                ( 6077.033,     30.385) ]


                let int64_stats = [   (  754.805,      6.793);
                                                ( 1932.444,     17.392);
                                                ( 2594.760,     23.353);
                                                ( 3340.833,     30.068);
                                                ( 4049.818,     36.448);
                                                ( 4265.048,     38.385);
                                                ( 4297.842,     38.681);
                                                ( 4304.978,     38.745);
                                                ( 4296.994,     38.673);
                                                ( 4272.609,     38.453) ]

                let float32_stats = [ (  700.645,      3.503);
                                                ( 1637.794,      8.189);
                                                ( 2532.372,     12.662);
                                                ( 3574.995,     17.875);
                                                ( 5125.039,     25.625);
                                                ( 6002.015,     30.010);
                                                ( 6084.097,     30.420);
                                                ( 6014.614,     30.073);
                                                ( 6001.517,     30.008);
                                                ( 5955.281,     29.776) ]

                let float64_stats = [ (  774.948,      6.975);
                                                ( 1903.411,     17.131);
                                                ( 2545.296,     22.908);
                                                ( 3361.132,     30.250);
                                                ( 4080.346,     36.723);
                                                ( 4274.057,     38.467);
                                                ( 4312.269,     38.810);
                                                ( 4320.384,     38.883);
                                                ( 4313.525,     38.822);
                                                ( 4287.927,     38.591) ]

                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]


            module SS2 =
                let notes = "Aka sorted search 2" // fix later
                let int32_stats = [  (  587.053,      4.696);
                                                ( 1435.379,     11.483);
                                                ( 2003.221,     16.026);
                                                ( 2655.954,     21.248);
                                                ( 3464.897,     27.719);
                                                ( 3845.386,     30.763);
                                                ( 3895.366,     31.163);
                                                ( 3864.740,     30.918);
                                                ( 3858.021,     30.864);
                                                ( 3837.459,     30.700) ]

                let int64_stats = [  (  617.302,      7.408);
                                                ( 1467.859,     17.614);
                                                ( 1987.696,     23.852);
                                                ( 2557.800,     30.694);
                                                ( 3032.869,     36.394);
                                                ( 3149.070,     37.789);
                                                ( 3172.130,     38.066);
                                                ( 3177.120,     38.125);
                                                ( 3175.214,     38.103);
                                                ( 3163.281,     37.959) ]

                let float32_stats = [(  588.608,      4.709);
                                                ( 1417.100,     11.337);
                                                ( 1980.612,     15.845);
                                                ( 2628.276,     21.026);
                                                ( 3383.959,     27.072);
                                                ( 3754.778,     30.038);
                                                ( 3812.464,     30.500);
                                                ( 3789.724,     30.318);
                                                ( 3782.860,     30.263);
                                                ( 3762.970,     30.104) ]

                let float64_stats = [(  619.803,      7.438);
                                                ( 1488.013,     17.856);
                                                ( 2015.431,     24.185);
                                                ( 2557.299,     30.688);
                                                ( 3032.727,     36.393);
                                                ( 3146.507,     37.758);
                                                ( 3178.239,     38.139);
                                                ( 3185.357,     38.224);
                                                ( 3185.774,     38.229);
                                                ( 3171.152,     38.054) ]
                
                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]


        module MergeStats =
            let tp = testParameters(Merge)
            let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations
            module Keys =
                let notes = ""
                let int32_stats = [  (  743.271,    5.946);
                                            ( 1517.431,   12.139);
                                            ( 2298.607,   18.389);
                                            ( 3508.165,   28.065);
                                            ( 5636.694,   45.094);
                                            ( 7263.641,   58.109);
                                            ( 7984.475,   63.876);
                                            ( 8017.395,   64.139);
                                            ( 7946.285,   63.570);
                                            ( 7877.881,   63.023) ]

                let int64_stats = [  (  724.760,   11.596);
                                            ( 1599.550,   25.593);
                                            ( 2183.630,   34.938);
                                            ( 2884.505,   46.152);
                                            ( 3540.747,   56.652);
                                            ( 3787.151,   60.594);
                                            ( 3934.479,   62.952);
                                            ( 3949.393,   63.190);
                                            ( 3936.593,   62.985);
                                            ( 3921.984,   62.752) ]

            module Pairs =
                let notes = ""
                let int32_stats = [ (  626.086,   10.017);
                                            ( 1317.770,   21.084);
                                            ( 1922.313,   30.757);
                                            ( 2788.777,   44.620);
                                            ( 4018.734,   64.300);
                                            ( 4535.611,   72.570);
                                            ( 4640.316,   74.245);
                                            ( 4593.082,   73.489);
                                            ( 4570.610,   73.130);
                                            ( 4525.218,   72.403) ]

                let int64_stats = [ (  629.958,   20.159);
                                            ( 1237.109,   39.587);
                                            ( 1648.341,   52.747);
                                            ( 2030.442,   64.974);
                                            ( 2355.326,   75.370);
                                            ( 2406.329,   77.003);
                                            ( 2426.950,   77.662);
                                            ( 2429.172,   77.734);
                                            ( 2414.210,   77.255);
                                            ( 2217.574,   70.962) ]

        module LoadBalanceStats =
            let tp = testParameters(LoadBalance)
            let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations
            module Regular =
                let notes = ""
                let int32_stats = [ (  895.387,   3.582);
                                        ( 2529.846,  10.119);
                                        ( 3545.643,  14.183);
                                        ( 4749.844,  18.999);
                                        ( 6451.189,  25.805);
                                        ( 7413.698,  29.655);
                                        ( 7841.852,  31.367);
                                        ( 7914.164,  31.657);
                                        ( 8006.468,  32.026);
                                        ( 7999.965,  32.000);
                                        ( 7987.870,  31.951) ]

            module ChangingRatio =
                let notes = "NumTerms ratio changed: ratio = 0.05 + 0.10 * ithTest"
                let int32_stats = [ ( 9549.279,  38.197);
                                        ( 8424.637,  33.699);
                                        ( 8009.964,  32.040);
                                        ( 7654.086,  30.616);
                                        ( 7402.806,  29.611);
                                        ( 7208.586,  28.834);
                                        ( 6912.631,  27.651);
                                        ( 6973.048,  27.892);
                                        ( 7066.564,  28.266);
                                        ( 7756.576,  31.026) ]

        module IntervalExpandStats =
            let tp = testParameters(IntervalMove)
            let sourceCounts, nIterations, termList = tp.SourceCounts, tp.Iterations, tp.Terms
            let constCounts = List.init termList.Length (fun _ -> 10000000)
            let constIterations = List.init termList.Length (fun _ -> 300)
            
            module AvgSegLength25 =
                let notes = "Average segment length of 25 elements"                

                let int32_stats = [  (  949.819,    4.103);
                                                                ( 2775.360,   11.990);
                                                                ( 3932.710,   16.989);
                                                                ( 4826.435,   20.850);
                                                                ( 5902.466,   25.499);
                                                                ( 6435.828,   27.803);
                                                                ( 6776.570,   29.275);
                                                                ( 6921.922,   29.903);
                                                                ( 7020.093,   30.327);
                                                                ( 7063.387,   30.514) ]
    
                let int64_stats = [  (  946.581,    8.027);
                                                                ( 2410.834,   20.444);
                                                                ( 3424.231,   29.037);
                                                                ( 4646.467,   39.402);
                                                                ( 5717.713,   48.486);
                                                                ( 6216.221,   52.714);
                                                                ( 6587.945,   55.866);
                                                                ( 6742.144,   57.173);
                                                                ( 6843.305,   58.031);
                                                                ( 6888.057,   58.411) ]

                let float32_stats = [    (  959.731,    4.146);
                                                                    ( 2855.482,   12.336);
                                                                    ( 3901.143,   16.853);
                                                                    ( 4837.809,   20.899);
                                                                    ( 5855.396,   25.295);
                                                                    ( 6365.269,   27.498);
                                                                    ( 6704.786,   28.965);
                                                                    ( 6876.665,   29.707);
                                                                    ( 6966.429,   30.095);
                                                                    ( 7011.741,   30.291) ]

                let float64_stats = [    (  965.414,    8.187);
                                                                    ( 2453.016,   20.802);
                                                                    ( 3444.643,   29.211);
                                                                    ( 4674.199,   39.637);
                                                                    ( 5695.453,   48.297);
                                                                    ( 6211.878,   52.677);
                                                                    ( 6588.935,   55.874);
                                                                    ( 6741.322,   57.166);
                                                                    ( 6845.326,   58.048);
                                                                    ( 6891.895,   58.443) ]
                                                                    
                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]
                         

            module ConstCountChangingExpandRate =
                let notes = "Constant 10M count and changing expand rate."                

                let int32_stats = [  ( 3986.239,   31.890);
                                                                        ( 5316.916,   29.775);
                                                                        ( 6145.892,   29.500);
                                                                        ( 6810.135,   29.965);
                                                                        ( 7744.778,   32.218);
                                                                        ( 8368.150,   34.142);
                                                                        ( 8819.202,   35.630);
                                                                        ( 9325.357,   37.451);
                                                                        ( 9702.836,   38.889);
                                                                        (10018.264,   40.113);
                                                                        (10252.709,   41.027) ]

                let int64_stats = [  ( 3815.821,   53.421);
                                                                        ( 5203.174,   54.113);
                                                                        ( 5886.311,   54.154);
                                                                        ( 6656.482,   57.246);
                                                                        ( 7479.449,   61.631);
                                                                        ( 8088.486,   65.679);
                                                                        ( 8558.418,   68.981);
                                                                        ( 9052.113,   72.634);
                                                                        ( 9388.806,   75.223);
                                                                        ( 9680.945,   77.506);
                                                                        ( 9965.576,   79.749) ]

                let float32_stats = [    ( 3962.179,   31.697);
                                                                            ( 5290.824,   29.629);
                                                                            ( 6101.382,   29.287);
                                                                            ( 6761.206,   29.749);
                                                                            ( 7666.630,   31.893);
                                                                            ( 8266.999,   33.729);
                                                                            ( 8700.010,   35.148);
                                                                            ( 9176.201,   36.852);
                                                                            ( 9529.223,   38.193);
                                                                            ( 9834.770,   39.378);
                                                                            (10059.946,   40.256) ]

                let float64_stats = [    ( 3816.959,   53.437);
                                                                            ( 5202.804,   54.109);
                                                                            ( 5887.199,   54.162);
                                                                            ( 6659.567,   57.272);
                                                                            ( 7478.098,   61.620);
                                                                            ( 8089.065,   65.683);
                                                                            ( 8559.188,   68.987);
                                                                            ( 9045.415,   72.580);
                                                                            ( 9387.526,   75.213);
                                                                            ( 9681.556,   77.511);
                                                                            ( 9965.176,   79.745) ]

                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]
//            let iexp_A_stats = [AvgSegLength25.int32_stats; AvgSegLength25.int64_stats; AvgSegLength25.float32_stats; AvgSegLength25.float64_stats]
//            let iexp_B_Stats = [ConstCountChangingExpandRate.int32_stats; ConstCountChangingExpandRate.int64_stats; ConstCountChangingExpandRate.float32_stats; ConstCountChangingExpandRate.float64_stats]


        module IntervalMoveStats =
            let tp = testParameters(IntervalMove)
            let sourceCounts, nIterations, termList = tp.SourceCounts, tp.Iterations, tp.Terms
            let constCounts = List.init termList.Length (fun _ -> 10000000)
            let constIterations = List.init termList.Length (fun _ -> 300)

            module AvgSegLength25 =
                let notes = "Average segment length of 25 elements."
                let sourceCounts, nIterations = tp.SourceCounts, tp.Iterations

                let int32_stats = [    (  842.971,    7.148);
                                                                ( 2204.996,   18.698);
                                                                ( 2909.897,   24.676);
                                                                ( 3434.157,   29.122);
                                                                ( 3988.186,   33.820);
                                                                ( 4203.390,   35.645);
                                                                ( 4352.266,   36.907);
                                                                ( 4425.224,   37.526);
                                                                ( 4466.706,   37.878);
                                                                ( 4456.259,   37.789) ]

                let int64_stats = [    (  811.530,   13.374);
                                                                ( 1759.138,   28.991);
                                                                ( 2390.025,   39.388);
                                                                ( 2820.412,   46.480);
                                                                ( 3211.733,   52.929);
                                                                ( 3374.127,   55.606);
                                                                ( 3478.124,   57.319);
                                                                ( 3517.098,   57.962);
                                                                ( 3518.465,   57.984);
                                                                ( 3510.597,   57.855) ]

                let float32_stats = [  (  845.023,    7.166);
                                                                ( 2180.694,   18.492);
                                                                ( 2906.350,   24.646);
                                                                ( 3448.409,   29.243);
                                                                ( 3964.375,   33.618);
                                                                ( 4197.930,   35.598);
                                                                ( 4354.251,   36.924);
                                                                ( 4427.241,   37.543);
                                                                ( 4466.232,   37.874);
                                                                ( 4455.616,   37.784) ]

                let float64_stats = [  (  809.344,   13.338);
                                                                ( 1740.078,   28.676);
                                                                ( 2394.072,   39.454);
                                                                ( 2815.788,   46.404);
                                                                ( 3209.556,   52.893);
                                                                ( 3372.701,   55.582);
                                                                ( 3478.075,   57.319);
                                                                ( 3519.011,   57.993);
                                                                ( 3518.377,   57.983);
                                                                ( 3510.142,   57.847) ]

                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]

            module ConstCountChangingExpandRate =
                let notes = "Constant 10M count and changing expand rate."                

                let int32_stats = [    (  755.823,   10.582);
                                                                        ( 1632.715,   16.980);
                                                                        ( 2832.459,   26.059);
                                                                        ( 4200.896,   36.128);
                                                                        ( 5104.535,   42.061);
                                                                        ( 5624.526,   45.671);
                                                                        ( 6014.156,   48.474);
                                                                        ( 6497.861,   52.139);
                                                                        ( 6810.674,   54.567);
                                                                        ( 7047.364,   56.421);
                                                                        ( 7233.359,   57.884) ]

                let int64_stats = [    (  727.324,   16.001);
                                                                        ( 1473.320,   27.109);
                                                                        ( 2344.295,   40.322);
                                                                        ( 3280.668,   54.459);
                                                                        ( 4055.152,   65.856);
                                                                        ( 4483.860,   72.280);
                                                                        ( 4784.601,   76.841);
                                                                        ( 5133.255,   82.255);
                                                                        ( 5386.205,   86.244);
                                                                        ( 5535.019,   88.594);
                                                                        ( 5631.035,   90.110) ]

                let float32_stats = [  (  756.540,   10.592);
                                                                        ( 1632.889,   16.982);
                                                                        ( 2826.309,   26.002);
                                                                        ( 4203.710,   36.152);
                                                                        ( 5102.682,   42.046);
                                                                        ( 5626.643,   45.688);
                                                                        ( 6014.625,   48.478);
                                                                        ( 6493.843,   52.107);
                                                                        ( 6813.264,   54.588);
                                                                        ( 7048.771,   56.432);
                                                                        ( 7231.932,   57.873) ]

                let float64_stats = [  (  727.406,   16.003);
                                                                        ( 1474.558,   27.132);
                                                                        ( 2340.767,   40.261);
                                                                        ( 3281.972,   54.481);
                                                                        ( 4054.164,   65.840);
                                                                        ( 4485.763,   72.310);
                                                                        ( 4781.396,   76.789);
                                                                        ( 5132.844,   82.249);
                                                                        ( 5383.327,   86.198);
                                                                        ( 5541.696,   88.700);
                                                                        ( 5643.895,   90.316) ]

                let fourTypeStatsList = [int32_stats; int64_stats; float32_stats; float64_stats]

//            let imv_A_stats = [AvgSegLength25.int32_stats; AvgSegLength25.int64_stats; AvgSegLength25.float32_stats; AvgSegLength25.float64_stats]
//            let imv_B_Stats = [ConstCountChangingExpandRate.int32_stats; ConstCountChangingExpandRate.int64_stats; ConstCountChangingExpandRate.float32_stats; ConstCountChangingExpandRate.float64_stats]


    module Thrust =
        module ScanStats =
            let notes = ""
            let int32_stats = [ (  26.547,     0.319);
                                 ( 123.302,     1.480); 
                                 ( 531.724,     6.381); 
                                 ( 971.429,    11.657);
                                 (1889.905,    22.679);
                                 (2313.313,    27.760);
                                 (3633.622,    43.603);
                                 (4492.701,    53.912);
                                 (4710.093,    56.521);
                                 (5088.089,    61.057) ]

            let int64_stats = [ (  62.907,     1.510); 
                                   ( 264.719,     6.353); 
                                   ( 467.517,    11.220); 
                                   ( 823.708,    19.769); 
                                   (1534.737,    36.834); 
                                   (2158.156,    51.796); 
                                   (2669.093,    64.058); 
                                   (3088.460,    74.123); 
                                   (3264.636,    78.351); 
                                   (3169.477,    76.067) ]