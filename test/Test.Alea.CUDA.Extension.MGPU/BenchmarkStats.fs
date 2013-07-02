﻿module Test.Alea.CUDA.Extension.MGPU.BenchmarkStats

// Sample stats from running various algorithm benchmarks.  Used for comparison


//Tesla K20c :  705.500 Mhz   (Ordinal 0)
//13 SMs enabled. Compute Capability sm_35
//FreeMem:   3576MB   TotalMem:   3584MB.
//Mem Clock: 2600.000 Mhz x 320 bits   (208.000 GB/s)
//ECC Enabled
module TeslaK20c =
                              // throughput,     bandwidth);
    let moderngpu_scanStats_int = [(  433.734,     5.205 );     // 10k
                                   (  702.083,     8.425 );     // 50k
                                   ( 1898.394,    22.781 );     // 100k
                                   ( 4527.857,    54.334 );     // 200k
                                   ( 7650.295,    91.804 );     // 500k
                                   ( 8946.772,   107.361 );     // 1M
                                   (10262.650,   123.152 );     // 2M
                                   (10547.469,   126.570 );     // 5M
                                   (10657.556,   127.891 );     // 10M
                                   (10627.220,   127.527 ) ]    // 20M

    let thrust_scanStats_int =  [ (    14.487,    0.174 );
                                  (   101.726,    1.221 );
                                  (   768.751,    9.225 );
                                  (  1480.935,   17.771 );
                                  (  2858.013,   34.296 );
                                  (  4125.783,   49.509 );
                                  (  4756.823,   57.082 );
                                  (  5575.663,   66.908 );
                                  (  5937.756,   71.253 );
                                  (  5985.076,   71.821 ) ]

    let moderngpu_scanStats_int64 = [ (   258.734,    6.210);      
                                      (   650.588,   15.614);     
                                      (  1206.063,   28.946);     
                                      (  2560.823,   61.460);     
                                      (  3972.426,   95.338);    
                                      (  4378.827,  105.092);    
                                      (  4725.657,  113.416);    
                                      (  4877.600,  117.062);    
                                      (  4920.150,  118.084);    
                                      (  4967.462,  119.219) ]   

    let thrust_scanStats_int64 =    [ (    64.414,    1.546);
                                      (   309.692,    7.433);
                                      (   520.199,   12.485);
                                      (   959.225,   23.021);
                                      (  2211.916,   53.086);
                                      (  2870.851,   68.900);
                                      (  3642.104,   87.411);
                                      (  4151.910,   99.646);
                                      (  4431.106,  106.347);
                                      (  4540.052,  108.961) ]


    let moderngpu_bulkRemoveStats_int = [(  183.576,      1.469);
                                         (  544.303,      4.354);
                                         ( 1248.267,      9.986);
                                         ( 3854.237,     30.834);
                                         ( 7095.487,     56.764);
                                         ( 9019.937,     72.159);
                                         (10351.320,     82.811);
                                         (11339.600,     90.717);
                                         (11649.941,     93.200);
                                         (11636.822,     93.095) ]
    

    let moderngpu_bulkRemoveStats_int64 = [ ( 341.621,      4.783);
                                            ( 759.194,     10.629);
                                            (1265.140,     17.712);
                                            (2290.602,     32.068);
                                            (6105.514,     85.477);
                                            (7570.339,    105.985);
                                            (8541.667,    119.583);
                                            (9256.332,    129.589);
                                            (9476.318,    132.668);
                                            (9473.003,    132.622) ]

    let moderngpu_bulkInsertStats_int = [ (   258.889,     2.589);
                                          (   514.024,     5.140);
                                          (  1720.477,    17.205);
                                          (  4132.535,    41.325);
                                          (  6701.114,    67.011);
                                          (  8462.207,    84.622);
                                          (  9559.348,    95.593);
                                          ( 10324.759,   103.248);
                                          ( 10290.732,   102.907);
                                          ( 10185.264,   101.853) ]

    let moderngpu_bulkInsertStats_int64 = [(  207.799,      3.740);
                                           (  721.625,     12.989);
                                           ( 2000.352,     36.006);
                                           ( 3082.903,     55.492);
                                           ( 5202.105,     93.638);
                                           ( 6197.645,    111.558);
                                           ( 6745.279,    121.415);
                                           ( 7083.957,    127.511);
                                           ( 7066.011,    127.188);
                                           ( 7034.488,    126.621) ]


//GeForce GTX 560 Ti : 1700.000 Mhz   (Ordinal 0)
//8 SMs enabled. Compute Capability sm_21
//FreeMem:    760MB   TotalMem:   1024MB.
//Mem Clock: 2004.000 Mhz x 256 bits   (128.256 GB/s)
//ECC Disabled
module GF560Ti = 
    let moderngpu_scanStats_int = [ ( 522.294,     6.268);    
                                    (1409.874,    16.918);     
                                    (2793.189,    33.518);     
                                    (5016.043,    60.193);     
                                    (7157.139,    85.886);     
                                    (7842.271,    94.107);     
                                    (8115.966,    97.392);     
                                    (8301.897,    99.623);     
                                    (8325.225,    99.903);     
                                    (8427.884,   101.135) ]

    let thrust_scanStats_int = [ (  26.547,     0.319);
                                 ( 123.302,     1.480); 
                                 ( 531.724,     6.381); 
                                 ( 971.429,    11.657);
                                 (1889.905,    22.679);
                                 (2313.313,    27.760);
                                 (3633.622,    43.603);
                                 (4492.701,    53.912);
                                 (4710.093,    56.521);
                                 (5088.089,    61.057) ]

    let moderngpu_scanStats_int64 = [ ( 364.793,     8.755);
                                      (1494.888,    35.877);  
                                      (2588.645,    62.127);  
                                      (3098.852,    74.372);  
                                      (3877.090,    93.050);  
                                      (4041.277,    96.991);  
                                      (4131.842,    99.164);  
                                      (4193.033,   100.633);  
                                      (4218.722,   101.249);  
                                      (4229.460,   101.507) ]


    let thrust_scanStats_int64 = [ (  62.907,     1.510); 
                                   ( 264.719,     6.353); 
                                   ( 467.517,    11.220); 
                                   ( 823.708,    19.769); 
                                   (1534.737,    36.834); 
                                   (2158.156,    51.796); 
                                   (2669.093,    64.058); 
                                   (3088.460,    74.123); 
                                   (3264.636,    78.351); 
                                   (3169.477,    76.067) ]


    let moderngpu_bulkRemoveStats_int = [  (  371.468,   2.972);
                                           ( 1597.495,  12.780);
                                           ( 3348.861,  26.791);
                                           ( 5039.794,  40.318);
                                           ( 7327.432,  58.619);
                                           ( 8625.687,  69.005);
                                           ( 9446.528,  75.572);
                                           ( 9877.425,  79.019);
                                           ( 9974.556,  79.796);
                                           (10060.556,  80.484)]


    let moderngpu_bulkRemoveStats_int64 = [ ( 328.193,  4.595);
                                            (1670.632, 23.389);
                                            (2898.674, 40.581);
                                            (3851.190, 53.917);
                                            (5057.443, 70.804);
                                            (5661.127, 79.256);
                                            (6052.202, 84.731);
                                            (6232.150, 87.250);
                                            (6273.645, 87.831);
                                            (6311.973, 88.638)]