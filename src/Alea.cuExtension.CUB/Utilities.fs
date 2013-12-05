(******************************************************************************
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and capable of managing device allocations on multiple devices.
 ******************************************************************************)

#pragma once

#ifndef __CUDA_ARCH__
    #include <set>              // NVCC (EDG, really) takes FOREVER to compile std::map
    #include <map>
#endif

#include <math.h>

#include "util_namespace.cuh"
#include "util_debug.cuh"

#include "host/spinlock.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


(**
 * \addtogroup UtilModule
 * @{
 *)


(******************************************************************************
 * CachingDeviceAllocator (host use)
 ******************************************************************************)

(**
 * \brief A simple caching allocator for device memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe and is capable of managing cached device allocations
 * on multiple devices.  It behaves as follows:
 *
 * \par
 * - Allocations categorized by bin size.
 * - Bin sizes progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused device allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations on a given device will exceed
 *   \p max_cached_bytes, allocations for that device are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingDeviceAllocator is configured with:
 * - \p bin_growth = 8
 * - \p min_bin = 3
 * - \p max_bin = 7
 * - \p max_cached_bytes = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per device
 *
 *)
struct CachingDeviceAllocator
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    enum
    {
        /// Invalid device ordinal
        INVALID_DEVICE_ORDINAL = -1,
    };

    (**
     * Integer pow function for unsigned base and exponent
     *)
    static unsigned int IntPow(
        unsigned int base,
        unsigned int exp)
    {
        unsigned int retval = 1;
        while (exp > 0)
        {
            if (exp & 1) {
                retval = retval * base;        // multiply the result by the current base
            }
            base = base * base;                // square the base
            exp = exp >> 1;                    // divide the exponent in half
        }
        return retval;
    }


    (**
     * Round up to the nearest power-of
     *)
    static void NearestPowerOf(
        unsigned int &power,
        size_t &rounded_bytes,
        unsigned int base,
        size_t value)
    {
        power = 0;
        rounded_bytes = 1;

        while (rounded_bytes < value)
        {
            rounded_bytes *= base;
            power++;
        }
    }

    (**
     * Descriptor for device memory allocations
     *)
    struct BlockDescriptor
    {
        int   device;        // device ordinal
        void*           d_ptr;      // Device pointer
        size_t          bytes;      // Size of allocation in bytes
        unsigned int    bin;        // Bin enumeration

        // Constructor
        BlockDescriptor(void *d_ptr, int device) :
            d_ptr(d_ptr),
            bytes(0),
            bin(0),
            device(device) {}

        // Constructor
        BlockDescriptor(size_t bytes, unsigned int bin, int device) :
            d_ptr(NULL),
            bytes(bytes),
            bin(bin),
            device(device) {}

        // Comparison functor for comparing device pointers
        static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.device < b.device) {
                return true;
            } else if (a.device > b.device) {
                return false;
            } else {
                return (a.d_ptr < b.d_ptr);
            }
        }

        // Comparison functor for comparing allocation sizes
        static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.device < b.device) {
                return true;
            } else if (a.device > b.device) {
                return false;
            } else {
                return (a.bytes < b.bytes);
            }
        }
    };

    /// BlockDescriptor comparator function interface
    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

#ifndef __CUDA_ARCH__   // Only define STL container members in host code

    /// Set type for cached blocks (ordered by size)
    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

    /// Set type for live blocks (ordered by ptr)
    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

    /// Map type of device ordinals to the number of cached bytes cached by each device
    typedef std::map<int, size_t> GpuCachedBytes;

#endif // __CUDA_ARCH__

    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    Spinlock        spin_lock;          /// Spinlock for thread-safety

    unsigned int    bin_growth;         /// Geometric growth factor for bin-sizes
    unsigned int    min_bin;            /// Minimum bin enumeration
    unsigned int    max_bin;            /// Maximum bin enumeration

    size_t          min_bin_bytes;      /// Minimum bin size
    size_t          max_bin_bytes;      /// Maximum bin size
    size_t          max_cached_bytes;   /// Maximum aggregate cached bytes per device

    bool            debug;              /// Whether or not to print (de)allocation events to stdout
    bool            skip_cleanup;       /// Whether or not to skip a call to FreeAllCached() when destructor is called.  (The CUDA runtime may have already shut down for statically declared allocators)

#ifndef __CUDA_ARCH__   // Only define STL container members in host code

    GpuCachedBytes  cached_bytes;       /// Map of device ordinal to aggregate cached bytes on that device
    CachedBlocks    cached_blocks;      /// Set of cached device allocations available for reuse
    BusyBlocks      live_blocks;        /// Set of live device allocations currently in use

#endif // __CUDA_ARCH__

#endif // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    (**
     * \brief Constructor.
     *)
    CachingDeviceAllocator(
        unsigned int bin_growth,    ///< Geometric growth factor for bin-sizes
        unsigned int min_bin,       ///< Minimum bin
        unsigned int max_bin,       ///< Maximum bin
        size_t max_cached_bytes)    ///< Maximum aggregate cached bytes per device
    :
    #ifndef __CUDA_ARCH__   // Only define STL container members in host code
            cached_blocks(BlockDescriptor::SizeCompare),
            live_blocks(BlockDescriptor::PtrCompare),
    #endif
            debug(false),
            spin_lock(0),
            bin_growth(bin_growth),
            min_bin(min_bin),
            max_bin(max_bin),
            min_bin_bytes(IntPow(bin_growth, min_bin)),
            max_bin_bytes(IntPow(bin_growth, max_bin)),
            max_cached_bytes(max_cached_bytes)
    {}


    (**
     * \brief Default constructor.
     *
     * Configured with:
     * \par
     * - \p bin_growth = 8
     * - \p min_bin = 3
     * - \p max_bin = 7
     * - \p max_cached_bytes = (\p bin_growth ^ \p max_bin) * 3) - 1 = 6,291,455 bytes
     *
     * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
     * sets a maximum of 6,291,455 cached bytes per device
     *)
    CachingDeviceAllocator(bool skip_cleanup = false) :
    #ifndef __CUDA_ARCH__   // Only define STL container members in host code
        cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare),
    #endif
        skip_cleanup(skip_cleanup),
        debug(false),
        spin_lock(0),
        bin_growth(8),
        min_bin(3),
        max_bin(7),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes((max_bin_bytes * 3) - 1)
    {}


    (**
     * \brief Sets the limit on the number bytes this allocator is allowed to cache per device.
     *)
    cudaError_t SetMaxCachedBytes(
        size_t max_cached_bytes)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        // Lock
        Lock(&spin_lock);

        this->max_cached_bytes = max_cached_bytes;

        if (debug) CubLog("New max_cached_bytes(%lld)\n", (long long) max_cached_bytes);

        // Unlock
        Unlock(&spin_lock);

        return cudaSuccess;

    #endif  // __CUDA_ARCH__
    }


    (**
     * \brief Provides a suitable allocation of device memory for the given size on the specified device
     *)
    cudaError_t DeviceAllocate(
        void** d_ptr,
        size_t bytes,
        int device)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        bool locked                     = false;
        int entrypoint_device           = INVALID_DEVICE_ORDINAL;
        cudaError_t error               = cudaSuccess;

        // Round up to nearest bin size
        unsigned int bin;
        size_t bin_bytes;
        NearestPowerOf(bin, bin_bytes, bin_growth, bytes);
        if (bin < min_bin) {
            bin = min_bin;
            bin_bytes = min_bin_bytes;
        }

        // Check if bin is greater than our maximum bin
        if (bin > max_bin)
        {
            // Allocate the request exactly and give out-of-range bin
            bin = (unsigned int) -1;
            bin_bytes = bytes;
        }

        BlockDescriptor search_key(bin_bytes, bin, device);

        // Lock
        if (!locked) {
            Lock(&spin_lock);
            locked = true;
        }

        do {
            // Find a free block big enough within the same bin on the same device
            CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
            if ((block_itr != cached_blocks.end()) &&
                (block_itr->device == device) &&
                (block_itr->bin == search_key.bin))
            {
                // Reuse existing cache block.  Insert into live blocks.
                search_key = *block_itr;
                live_blocks.insert(search_key);

                // Remove from free blocks
                cached_blocks.erase(block_itr);
                cached_bytes[device] -= search_key.bytes;

                if (debug) CubLog("\tdevice %d reused cached block (%lld bytes). %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                    device, (long long) search_key.bytes, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
            }
            else
            {
                // Need to allocate a new cache block. Unlock.
                if (locked) {
                    Unlock(&spin_lock);
                    locked = false;
                }

                // Set to specified device
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
                if (CubDebug(error = cudaSetDevice(device))) break;

                // Allocate
                if (CubDebug(error = cudaMalloc(&search_key.d_ptr, search_key.bytes))) break;

                // Lock
                if (!locked) {
                    Lock(&spin_lock);
                    locked = true;
                }

                // Insert into live blocks
                live_blocks.insert(search_key);

                if (debug) CubLog("\tdevice %d allocating new device block %lld bytes. %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                    device, (long long) search_key.bytes, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
            }
        } while(0);

        // Unlock
        if (locked) {
            Unlock(&spin_lock);
            locked = false;
        }

        // Copy device pointer to output parameter (NULL on error)
        *d_ptr = search_key.d_ptr;

        // Attempt to revert back to previous device if necessary
        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
        {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;

    #endif  // __CUDA_ARCH__
    }


    (**
     * \brief Provides a suitable allocation of device memory for the given size on the current device
     *)
    cudaError_t DeviceAllocate(
        void** d_ptr,
        size_t bytes)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else
        cudaError_t error = cudaSuccess;
        do {
            int current_device;
            if (CubDebug(error = cudaGetDevice(&current_device))) break;
            if (CubDebug(error = DeviceAllocate(d_ptr, bytes, current_device))) break;
        } while(0);

        return error;

    #endif  // __CUDA_ARCH__
    }


    (**
     * \brief Frees a live allocation of device memory on the specified device, returning it to the allocator
     *)
    cudaError_t DeviceFree(
        void* d_ptr,
        int device)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        bool locked                     = false;
        int entrypoint_device           = INVALID_DEVICE_ORDINAL;
        cudaError_t error               = cudaSuccess;

        BlockDescriptor search_key(d_ptr, device);

        // Lock
        if (!locked) {
            Lock(&spin_lock);
            locked = true;
        }

        do {
            // Find corresponding block descriptor
            BusyBlocks::iterator block_itr = live_blocks.find(search_key);
            if (block_itr == live_blocks.end())
            {
                // Cannot find pointer
                if (CubDebug(error = cudaErrorUnknown)) break;
            }
            else
            {
                // Remove from live blocks
                search_key = *block_itr;
                live_blocks.erase(block_itr);

                // Check if we should keep the returned allocation
                if (cached_bytes[device] + search_key.bytes <= max_cached_bytes)
                {
                    // Insert returned allocation into free blocks
                    cached_blocks.insert(search_key);
                    cached_bytes[device] += search_key.bytes;

                    if (debug) CubLog("\tdevice %d returned %lld bytes. %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                        device, (long long) search_key.bytes, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
                }
                else
                {
                    // Free the returned allocation.  Unlock.
                    if (locked) {
                        Unlock(&spin_lock);
                        locked = false;
                    }

                    // Set to specified device
                    if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
                    if (CubDebug(error = cudaSetDevice(device))) break;

                    // Free device memory
                    if (CubDebug(error = cudaFree(d_ptr))) break;

                    if (debug) CubLog("\tdevice %d freed %lld bytes.  %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                        device, (long long) search_key.bytes, (long long) cached_blocks.size(), (long long) cached_bytes[device], (long long) live_blocks.size());
                }
            }
        } while (0);

        // Unlock
        if (locked) {
            Unlock(&spin_lock);
            locked = false;
        }

        // Attempt to revert back to entry-point device if necessary
        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
        {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;

    #endif  // __CUDA_ARCH__
    }


    (**
     * \brief Frees a live allocation of device memory on the current device, returning it to the allocator
     *)
    cudaError_t DeviceFree(
        void* d_ptr)
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        int current_device;
        cudaError_t error = cudaSuccess;

        do {
            if (CubDebug(error = cudaGetDevice(&current_device))) break;
            if (CubDebug(error = DeviceFree(d_ptr, current_device))) break;
        } while(0);

        return error;

    #endif  // __CUDA_ARCH__
    }


    (**
     * \brief Frees all cached device allocations on all devices
     *)
    cudaError_t FreeAllCached()
    {
    #ifdef __CUDA_ARCH__
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        cudaError_t error         = cudaSuccess;
        bool locked               = false;
        int entrypoint_device     = INVALID_DEVICE_ORDINAL;
        int current_device        = INVALID_DEVICE_ORDINAL;

        // Lock
        if (!locked) {
            Lock(&spin_lock);
            locked = true;
        }

        while (!cached_blocks.empty())
        {
            // Get first block
            CachedBlocks::iterator begin = cached_blocks.begin();

            // Get entry-point device ordinal if necessary
            if (entrypoint_device == INVALID_DEVICE_ORDINAL)
            {
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
            }

            // Set current device ordinal if necessary
            if (begin->device != current_device)
            {
                if (CubDebug(error = cudaSetDevice(begin->device))) break;
                current_device = begin->device;
            }

            // Free device memory
            if (CubDebug(error = cudaFree(begin->d_ptr))) break;

            // Reduce balance and erase entry
            cached_bytes[current_device] -= begin->bytes;
            cached_blocks.erase(begin);

            if (debug) CubLog("\tdevice %d freed %lld bytes.  %lld available blocks cached (%lld bytes), %lld live blocks outstanding.\n",
                current_device, (long long) begin->bytes, (long long) cached_blocks.size(), (long long) cached_bytes[current_device], (long long) live_blocks.size());
        }

        // Unlock
        if (locked) {
            Unlock(&spin_lock);
            locked = false;
        }

        // Attempt to revert back to entry-point device if necessary
        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
        {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;

    #endif  // __CUDA_ARCH__
    }


    (**
     * \brief Destructor
     *)
    virtual ~CachingDeviceAllocator()
    {
        if (!skip_cleanup)
            FreeAllCached();
    }

};




(** @} *)       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


(**
 * \file
 * Static architectural properties by SM version.
 *)


(******************************************************************************
 * Static architectural properties by SM version.
 *
 * "Device" reflects the PTX architecture targeted by the active compiler
 * pass.  It provides useful compile-time statics within device code.  E.g.,:
 *
 *     __shared__ int[Device::WARP_THREADS];
 *
 *     int padded_offset = threadIdx.x + (threadIdx.x >> Device::LOG_SMEM_BANKS);
 *
 ******************************************************************************)

#pragma once

#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


(**
 * \addtogroup UtilModule
 * @{
 *)


/// CUB_PTX_ARCH reflects the PTX version targeted by the active compiler pass (or zero during the host pass).
#ifndef __CUDA_ARCH__
    #define CUB_PTX_ARCH 0
#else
    #define CUB_PTX_ARCH __CUDA_ARCH__
#endif


/// Whether or not the source targeted by the active compiler pass is allowed to  invoke device kernels or methods from the CUDA runtime API.
#if !defined(__CUDA_ARCH__) || defined(CUB_CDP)
#define CUB_RUNTIME_ENABLED
#endif


/// Execution space for destructors
#if ((CUB_PTX_ARCH > 0) && (CUB_PTX_ARCH < 200))
    #define CUB_DESTRUCTOR __host__
#else
    #define CUB_DESTRUCTOR __host__ __device__
#endif


(**
 * \brief Structure for statically reporting CUDA device properties, parameterized by SM architecture.
 *
 * The default specialization is for SM10.
 *)
template <int SM_ARCH>
struct ArchProps
{
    enum
    {
        LOG_WARP_THREADS    =
                                        5,                        /// Log of the number of threads per warp
        WARP_THREADS        =
                                        1 << LOG_WARP_THREADS,    /// Number of threads per warp
        LOG_SMEM_BANKS      =
                                        4,                        /// Log of the number of smem banks
        SMEM_BANKS          =
                                        1 << LOG_SMEM_BANKS,      /// The number of smem banks
        SMEM_BANK_BYTES     =
                                        4,                        /// Size of smem bank words
        SMEM_BYTES          =
                                        16 * 1024,                /// Maximum SM shared memory
        SMEM_ALLOC_UNIT     =
                                        512,                      /// Smem allocation size in bytes
        REGS_BY_BLOCK       =
                                        true,                     /// Whether or not the architecture allocates registers by block (or by warp)
        REG_ALLOC_UNIT      =
                                        256,                      /// Number of registers allocated at a time per block (or by warp)
        WARP_ALLOC_UNIT     =
                                        2,                        /// Granularity of warps for which registers are allocated
        MAX_SM_THREADS      =
                                        768,                      /// Maximum number of threads per SM
        MAX_SM_THREADBLOCKS =
                                        8,                        /// Maximum number of thread blocks per SM
        MAX_BLOCK_THREADS   =
                                        512,                      /// Maximum number of thread per thread block
        MAX_SM_REGISTERS    =
                                        8 * 1024,                 /// Maximum number of registers per SM
    };
};




#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

(**
 * Architecture properties for SM30
 *)
template <>
struct ArchProps<300>
{
    enum
    {
        LOG_WARP_THREADS    = 5,                        // 32 threads per warp
        WARP_THREADS        = 1 << LOG_WARP_THREADS,
        LOG_SMEM_BANKS      = 5,                        // 32 banks
        SMEM_BANKS          = 1 << LOG_SMEM_BANKS,
        SMEM_BANK_BYTES     = 4,                        // 4 byte bank words
        SMEM_BYTES          = 48 * 1024,                // 48KB shared memory
        SMEM_ALLOC_UNIT     = 256,                      // 256B smem allocation segment size
        REGS_BY_BLOCK       = false,                    // Allocates registers by warp
        REG_ALLOC_UNIT      = 256,                      // 256 registers allocated at a time per warp
        WARP_ALLOC_UNIT     = 4,                        // Registers are allocated at a granularity of every 4 warps per threadblock
        MAX_SM_THREADS      = 2048,                     // 2K max threads per SM
        MAX_SM_THREADBLOCKS = 16,                       // 16 max threadblocks per SM
        MAX_BLOCK_THREADS   = 1024,                     // 1024 max threads per threadblock
        MAX_SM_REGISTERS    = 64 * 1024,                // 64K max registers per SM
    };

    // Callback utility
    template <typename T>
    static __host__ __device__ __forceinline__ void Callback(T &target, int sm_version)
    {
        target.template Callback<ArchProps>();
    }
};


(**
 * Architecture properties for SM20
 *)
template <>
struct ArchProps<200>
{
    enum
    {
        LOG_WARP_THREADS    = 5,                        // 32 threads per warp
        WARP_THREADS        = 1 << LOG_WARP_THREADS,
        LOG_SMEM_BANKS      = 5,                        // 32 banks
        SMEM_BANKS          = 1 << LOG_SMEM_BANKS,
        SMEM_BANK_BYTES     = 4,                        // 4 byte bank words
        SMEM_BYTES          = 48 * 1024,                // 48KB shared memory
        SMEM_ALLOC_UNIT     = 128,                      // 128B smem allocation segment size
        REGS_BY_BLOCK       = false,                    // Allocates registers by warp
        REG_ALLOC_UNIT      = 64,                       // 64 registers allocated at a time per warp
        WARP_ALLOC_UNIT     = 2,                        // Registers are allocated at a granularity of every 2 warps per threadblock
        MAX_SM_THREADS      = 1536,                     // 1536 max threads per SM
        MAX_SM_THREADBLOCKS = 8,                        // 8 max threadblocks per SM
        MAX_BLOCK_THREADS   = 1024,                     // 1024 max threads per threadblock
        MAX_SM_REGISTERS    = 32 * 1024,                // 32K max registers per SM
    };

    // Callback utility
    template <typename T>
    static __host__ __device__ __forceinline__ void Callback(T &target, int sm_version)
    {
        if (sm_version > 200) {
            ArchProps<300>::Callback(target, sm_version);
        } else {
            target.template Callback<ArchProps>();
        }
    }
};


(**
 * Architecture properties for SM12
 *)
template <>
struct ArchProps<120>
{
    enum
    {
        LOG_WARP_THREADS    = 5,                        // 32 threads per warp
        WARP_THREADS        = 1 << LOG_WARP_THREADS,
        LOG_SMEM_BANKS      = 4,                        // 16 banks
        SMEM_BANKS          = 1 << LOG_SMEM_BANKS,
        SMEM_BANK_BYTES     = 4,                        // 4 byte bank words
        SMEM_BYTES          = 16 * 1024,                // 16KB shared memory
        SMEM_ALLOC_UNIT     = 512,                      // 512B smem allocation segment size
        REGS_BY_BLOCK       = true,                     // Allocates registers by threadblock
        REG_ALLOC_UNIT      = 512,                      // 512 registers allocated at time per threadblock
        WARP_ALLOC_UNIT     = 2,                        // Registers are allocated at a granularity of every 2 warps per threadblock
        MAX_SM_THREADS      = 1024,                     // 1024 max threads per SM
        MAX_SM_THREADBLOCKS = 8,                        // 8 max threadblocks per SM
        MAX_BLOCK_THREADS   = 512,                      // 512 max threads per threadblock
        MAX_SM_REGISTERS    = 16 * 1024,                // 16K max registers per SM
    };

    // Callback utility
    template <typename T>
    static __host__ __device__ __forceinline__ void Callback(T &target, int sm_version)
    {
        if (sm_version > 120) {
            ArchProps<200>::Callback(target, sm_version);
        } else {
            target.template Callback<ArchProps>();
        }
    }
};


(**
 * Architecture properties for SM10.  Derives from the default ArchProps specialization.
 *)
template <>
struct ArchProps<100> : ArchProps<0>
{
    // Callback utility
    template <typename T>
    static __host__ __device__ __forceinline__ void Callback(T &target, int sm_version)
    {
        if (sm_version > 100) {
            ArchProps<120>::Callback(target, sm_version);
        } else {
            target.template Callback<ArchProps>();
        }
    }
};


(**
 * Architecture properties for SM35
 *)
template <>
struct ArchProps<350> : ArchProps<300> {};        // Derives from SM30

(**
 * Architecture properties for SM21
 *)
template <>
struct ArchProps<210> : ArchProps<200> {};        // Derives from SM20

(**
 * Architecture properties for SM13
 *)
template <>
struct ArchProps<130> : ArchProps<120> {};        // Derives from SM12

(**
 * Architecture properties for SM11
 *)
template <>
struct ArchProps<110> : ArchProps<100> {};        // Derives from SM10


#endif // DOXYGEN_SHOULD_SKIP_THIS


(**
 * \brief The architectural properties for the PTX version targeted by the active compiler pass.
 *)
struct PtxArchProps : ArchProps<CUB_PTX_ARCH> {};


(** @} *)       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


(**
 * \file
 * Error and event logging routines.
 *
 * The following macros definitions are supported:
 * - \p CUB_LOG.  Simple event messages are printed to \p stdout.
 *)

#pragma once

#include <stdio.h>
#include "util_namespace.cuh"
#include "util_arch.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


(**
 * \addtogroup UtilModule
 * @{
 *)


/// CUB error reporting macro (prints error messages to stderr)
#if (defined(DEBUG) || defined(_DEBUG))
    #define CUB_STDERR
#endif



(**
 * \brief %If \p CUB_STDERR is defined and \p error is not \p cudaSuccess, the corresponding error message is printed to \p stderr (or \p stdout in device code) along with the supplied source context.
 *
 * \return The CUDA error.
 *)
__host__ __device__ __forceinline__ cudaError_t Debug(
    cudaError_t     error,
    const char*     filename,
    int             line)
{
#ifdef CUB_STDERR
    if (error)
    {
    #if (CUB_PTX_ARCH == 0)
        fprintf(stderr, "CUDA error %d [%s, %d]: %s\n", error, filename, line, cudaGetErrorString(error));
        fflush(stderr);
    #elif (CUB_PTX_ARCH >= 200)
        printf("CUDA error %d [block %d, thread %d, %s, %d]\n", error, blockIdx.x, threadIdx.x, filename, line);
    #endif
    }
#endif
    return error;
}


(**
 * \brief Debug macro
 *)
#define CubDebug(e) cub::Debug((e), __FILE__, __LINE__)


(**
 * \brief Debug macro with exit
 *)
#define CubDebugExit(e) if (cub::Debug((e), __FILE__, __LINE__)) { exit(1); }


(**
 * \brief Log macro for printf statements.
 *)
#if (CUB_PTX_ARCH == 0)
    #define CubLog(format, ...) printf(format,__VA_ARGS__);
#elif (CUB_PTX_ARCH >= 200)
    #define CubLog(format, ...) printf("[block %d, thread %d]: " format, blockIdx.x, threadIdx.x, __VA_ARGS__);
#endif




(** @} *)       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

(**
 * \file
 * Properties of a given CUDA device and the corresponding PTX bundle
 *)

#pragma once

#include "util_arch.cuh"
#include "util_debug.cuh"
#include "util_namespace.cuh"
#include "util_macro.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


(**
 * \addtogroup UtilModule
 * @{
 *)

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


(**
 * Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 *)
template <typename T>
__global__ void EmptyKernel(void) { }


(**
 * Alias temporaries to externally-allocated device storage (or simply return the amount of storage needed).
 *)
template <int ALLOCATIONS>
__host__ __device__ __forceinline__
cudaError_t AliasTemporaries(
    void    *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
    size_t  &temp_storage_bytes,                ///< [in,out] Size in bytes of \t d_temp_storage allocation
    void*   (&allocations)[ALLOCATIONS],        ///< [in,out] Pointers to device allocations needed
    size_t  (&allocation_sizes)[ALLOCATIONS])   ///< [in] Sizes in bytes of device allocations needed
{
    const int ALIGN_BYTES   = 256;
    const int ALIGN_MASK    = ~(ALIGN_BYTES - 1);

    // Compute exclusive prefix sum over allocation requests
    size_t bytes_needed = 0;
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        size_t allocation_bytes = (allocation_sizes[i] + ALIGN_BYTES - 1) & ALIGN_MASK;
        allocation_sizes[i] = bytes_needed;
        bytes_needed += allocation_bytes;
    }

    // Check if the caller is simply requesting the size of the storage allocation
    if (!d_temp_storage)
    {
        temp_storage_bytes = bytes_needed;
        return cudaSuccess;
    }

    // Check if enough storage provided
    if (temp_storage_bytes < bytes_needed)
    {
        return CubDebug(cudaErrorMemoryAllocation);
    }

    // Alias
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        allocations[i] = static_cast<char*>(d_temp_storage) + allocation_sizes[i];
    }

    return cudaSuccess;
}



#endif  // DOXYGEN_SHOULD_SKIP_THIS



(**
 * \brief Retrieves the PTX version (major * 100 + minor * 10)
 *)
__host__ __device__ __forceinline__ cudaError_t PtxVersion(int &ptx_version)
{
#ifndef CUB_RUNTIME_ENABLED

    // CUDA API calls not supported from this device
    return cudaErrorInvalidConfiguration;

#else

    cudaError_t error = cudaSuccess;
    do
    {
        cudaFuncAttributes empty_kernel_attrs;
        if (CubDebug(error = cudaFuncGetAttributes(&empty_kernel_attrs, EmptyKernel<void>))) break;
        ptx_version = empty_kernel_attrs.ptxVersion * 10;
    }
    while (0);

    return error;

#endif
}


(**
 * Synchronize the stream if specified
 *)
__host__ __device__ __forceinline__
static cudaError_t SyncStream(cudaStream_t stream)
{
#ifndef __CUDA_ARCH__
    return cudaStreamSynchronize(stream);
#else
    // Device can't yet sync on a specific stream
    return cudaDeviceSynchronize();
#endif
}



(**
 * \brief Properties of a given CUDA device and the corresponding PTX bundle
 *)
class Device
{
private:

    /// Type definition of the EmptyKernel kernel entry point
    typedef void (*EmptyKernelPtr)();

    /// Force EmptyKernel<void> to be generated if this class is used
    __host__ __device__ __forceinline__
    EmptyKernelPtr Empty()
    {
        return EmptyKernel<void>;
    }

public:

    // Version information
    int     sm_version;             ///< SM version of target device (SM version X.YZ in XYZ integer form)
    int     ptx_version;            ///< Bundled PTX version for target device (PTX version X.YZ in XYZ integer form)

    // Target device properties
    int     sm_count;               ///< Number of SMs
    int     warp_threads;           ///< Number of threads per warp
    int     smem_bank_bytes;        ///< Number of bytes per SM bank
    int     smem_banks;             ///< Number of smem banks
    int     smem_bytes;             ///< Smem bytes per SM
    int     smem_alloc_unit;        ///< Smem segment size
    bool    regs_by_block;          ///< Whether registers are allocated by threadblock (or by warp)
    int     reg_alloc_unit;         ///< Granularity of register allocation within the SM
    int     warp_alloc_unit;        ///< Granularity of warp allocation within the SM
    int     max_sm_threads;         ///< Maximum number of threads per SM
    int     max_sm_blocks;          ///< Maximum number of threadblocks per SM
    int     max_block_threads;      ///< Maximum number of threads per threadblock
    int     max_sm_registers;       ///< Maximum number of registers per SM
    int     max_sm_warps;           ///< Maximum number of warps per SM

    (**
     * Callback for initializing device properties
     *)
    template <typename ArchProps>
    __host__ __device__ __forceinline__ void Callback()
    {
        warp_threads        = ArchProps::WARP_THREADS;
        smem_bank_bytes     = ArchProps::SMEM_BANK_BYTES;
        smem_banks          = ArchProps::SMEM_BANKS;
        smem_bytes          = ArchProps::SMEM_BYTES;
        smem_alloc_unit     = ArchProps::SMEM_ALLOC_UNIT;
        regs_by_block       = ArchProps::REGS_BY_BLOCK;
        reg_alloc_unit      = ArchProps::REG_ALLOC_UNIT;
        warp_alloc_unit     = ArchProps::WARP_ALLOC_UNIT;
        max_sm_threads      = ArchProps::MAX_SM_THREADS;
        max_sm_blocks       = ArchProps::MAX_SM_THREADBLOCKS;
        max_block_threads   = ArchProps::MAX_BLOCK_THREADS;
        max_sm_registers    = ArchProps::MAX_SM_REGISTERS;
        max_sm_warps        = max_sm_threads / warp_threads;
    }


public:

    (**
     * Initializer.  Properties are retrieved for the specified GPU ordinal.
     *)
    __host__ __device__ __forceinline__
    cudaError_t Init(int device_ordinal)
    {
    #ifndef CUB_RUNTIME_ENABLED

        // CUDA API calls not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);

    #else

        cudaError_t error = cudaSuccess;
        do
        {
            // Fill in SM version
            int major, minor;
            if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
            if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
            sm_version = major * 100 + minor * 10;

            // Fill in static SM properties
            // Initialize our device properties via callback from static device properties
            ArchProps<100>::Callback(*this, sm_version);

            // Fill in SM count
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Fill in PTX version
        #if CUB_PTX_ARCH > 0
            ptx_version = CUB_PTX_ARCH;
        #else
            if (CubDebug(error = PtxVersion(ptx_version))) break;
        #endif

        }
        while (0);

        return error;

    #endif
    }


    (**
     * Initializer.  Properties are retrieved for the current GPU ordinal.
     *)
    __host__ __device__ __forceinline__
    cudaError_t Init()
    {
    #ifndef CUB_RUNTIME_ENABLED

        // CUDA API calls not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);

    #else

        cudaError_t error = cudaSuccess;
        do
        {
            int device_ordinal;
            if ((error = CubDebug(cudaGetDevice(&device_ordinal)))) break;
            if ((error = Init(device_ordinal))) break;
        }
        while (0);
        return error;

    #endif
    }


    (**
     * Computes maximum SM occupancy in thread blocks for the given kernel
     *)
    template <typename KernelPtr>
    __host__ __device__ __forceinline__
    cudaError_t MaxSmOccupancy(
        int                 &max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
        KernelPtr           kernel_ptr,                 ///< [in] Kernel pointer for which to compute SM occupancy
        int                 block_threads)              ///< [in] Number of threads per thread block
    {
    #ifndef CUB_RUNTIME_ENABLED

        // CUDA API calls not supported from this device
        return CubDebug(cudaErrorInvalidConfiguration);

    #else

        cudaError_t error = cudaSuccess;
        do
        {
            // Get kernel attributes
            cudaFuncAttributes kernel_attrs;
            if (CubDebug(error = cudaFuncGetAttributes(&kernel_attrs, kernel_ptr))) break;

            // Number of warps per threadblock
            int block_warps = (block_threads +  warp_threads - 1) / warp_threads;

            // Max warp occupancy
            int max_warp_occupancy = (block_warps > 0) ?
                max_sm_warps / block_warps :
                max_sm_blocks;

            // Maximum register occupancy
            int max_reg_occupancy;
            if ((block_threads == 0) || (kernel_attrs.numRegs == 0))
            {
                // Prevent divide-by-zero
                max_reg_occupancy = max_sm_blocks;
            }
            else if (regs_by_block)
            {
                // Allocates registers by threadblock
                int block_regs = CUB_ROUND_UP_NEAREST(kernel_attrs.numRegs * warp_threads * block_warps, reg_alloc_unit);
                max_reg_occupancy = max_sm_registers / block_regs;
            }
            else
            {
                // Allocates registers by warp
                int sm_sides                = warp_alloc_unit;
                int sm_registers_per_side   = max_sm_registers / sm_sides;
                int regs_per_warp           = CUB_ROUND_UP_NEAREST(kernel_attrs.numRegs * warp_threads, reg_alloc_unit);
                int warps_per_side          = sm_registers_per_side / regs_per_warp;
                int warps                   = warps_per_side * sm_sides;
                max_reg_occupancy           = warps / block_warps;
            }

            // Shared memory per threadblock
            int block_allocated_smem = CUB_ROUND_UP_NEAREST(
                kernel_attrs.sharedSizeBytes,
                smem_alloc_unit);

            // Max shared memory occupancy
            int max_smem_occupancy = (block_allocated_smem > 0) ?
                (smem_bytes / block_allocated_smem) :
                max_sm_blocks;

            // Max occupancy
            max_sm_occupancy = CUB_MIN(
                CUB_MIN(max_sm_blocks, max_warp_occupancy),
                CUB_MIN(max_smem_occupancy, max_reg_occupancy));

//            printf("max_smem_occupancy(%d), max_warp_occupancy(%d), max_reg_occupancy(%d)", max_smem_occupancy, max_warp_occupancy, max_reg_occupancy);

        } while (0);

        return error;

    #endif
    }

};


(** @} *)       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

(**
 * \file
 * Random-access iterator types
 *)

#pragma once

#include "thread/thread_load.cuh"
#include "util_device.cuh"
#include "util_debug.cuh"
#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


(******************************************************************************
 * Texture references
 *****************************************************************************)

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

// Anonymous namespace
namespace {

/// Templated texture reference type
template <typename T>
struct TexIteratorRef
{
    // Texture reference type
    typedef texture<T, cudaTextureType1D, cudaReadModeElementType> TexRef;

    static TexRef ref;

    (**
     * Bind texture
     *)
    static cudaError_t BindTexture(void *d_in)
    {
        cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<T>();
        if (d_in)
            return (CubDebug(cudaBindTexture(NULL, ref, d_in, tex_desc)));

        return cudaSuccess;
    }

    (**
     * Unbind textures
     *)
    static cudaError_t UnbindTexture()
    {
        return CubDebug(cudaUnbindTexture(ref));
    }
};

// Texture reference definitions
template <typename Value>
typename TexIteratorRef<Value>::TexRef TexIteratorRef<Value>::ref = 0;

} // Anonymous namespace


#endif // DOXYGEN_SHOULD_SKIP_THIS







(**
 * \addtogroup UtilModule
 * @{
 *)


(******************************************************************************
 * Iterators
 *****************************************************************************)

(**
 * \brief A simple random-access iterator pointing to a range of constant values
 *
 * \par Overview
 * ConstantIteratorRA is a random-access iterator that when dereferenced, always
 * returns the supplied constant of type \p OutputType.
 *
 * \tparam OutputType           The value type of this iterator
 *)
template <typename OutputType>
class ConstantIteratorRA
{
public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    typedef ConstantIteratorRA                  self_type;
    typedef OutputType                          value_type;
    typedef OutputType                          reference;
    typedef OutputType*                         pointer;
    typedef std::random_access_iterator_tag     iterator_category;
    typedef int                                 difference_type;

#endif  // DOXYGEN_SHOULD_SKIP_THIS

private:

    OutputType    val;

public:

    /// Constructor
    __host__ __device__ __forceinline__ ConstantIteratorRA(
        const OutputType &val)          ///< Constant value for the iterator instance to report
    :
        val(val)
    {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    __host__ __device__ __forceinline__ self_type operator++()
    {
        self_type i = *this;
        return i;
    }

    __host__ __device__ __forceinline__ self_type operator++(int junk)
    {
        return *this;
    }

    __host__ __device__ __forceinline__ reference operator*()
    {
        return val;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator+(SizeT n)
    {
        return ConstantIteratorRA(val);
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator-(SizeT n)
    {
        return ConstantIteratorRA(val);
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ reference operator[](SizeT n)
    {
        return ConstantIteratorRA(val);
    }

    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &val;
    }

    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (val == rhs.val);
    }

    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (val != rhs.val);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

};



(**
 * \brief A simple random-access transform iterator for applying a transformation operator.
 *
 * \par Overview
 * TransformIteratorRA is a random-access iterator that wraps both a native
 * device pointer of type <tt>InputType*</tt> and a unary conversion functor of
 * type \p ConversionOp. \p OutputType references are made by pulling \p InputType
 * values through the \p ConversionOp instance.
 *
 * \tparam InputType            The value type of the pointer being wrapped
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p OutputType.  Must have member <tt>OutputType operator()(const InputType &datum)</tt>.
 * \tparam OutputType           The value type of this iterator
 *)
template <typename OutputType, typename ConversionOp, typename InputType>
class TransformIteratorRA
{
public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    typedef TransformIteratorRA                 self_type;
    typedef OutputType                          value_type;
    typedef OutputType                          reference;
    typedef OutputType*                         pointer;
    typedef std::random_access_iterator_tag     iterator_category;
    typedef int                                 difference_type;

#endif  // DOXYGEN_SHOULD_SKIP_THIS

private:

    ConversionOp    conversion_op;
    InputType*      ptr;

public:

    (**
     * \brief Constructor
     * @param ptr Native pointer to wrap
     * @param conversion_op Binary transformation functor
     *)
    __host__ __device__ __forceinline__ TransformIteratorRA(InputType* ptr, ConversionOp conversion_op) :
        conversion_op(conversion_op),
        ptr(ptr) {}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    __host__ __device__ __forceinline__ self_type operator++()
    {
        self_type i = *this;
        ptr++;
        return i;
    }

    __host__ __device__ __forceinline__ self_type operator++(int junk)
    {
        ptr++;
        return *this;
    }

    __host__ __device__ __forceinline__ reference operator*()
    {
        return conversion_op(*ptr);
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator+(SizeT n)
    {
        TransformIteratorRA retval(ptr + n, conversion_op);
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator-(SizeT n)
    {
        TransformIteratorRA retval(ptr - n, conversion_op);
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ reference operator[](SizeT n)
    {
        return conversion_op(ptr[n]);
    }

    __host__ __device__ __forceinline__ pointer operator->()
    {
        return &conversion_op(*ptr);
    }

    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }

    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

};



(**
 * \brief A simple random-access iterator for loading primitive values through texture cache.
 *
 * \par Overview
 * TexIteratorRA is a random-access iterator that wraps a native
 * device pointer of type <tt>T*</tt>. References made through TexIteratorRA
 * causes values to be pulled through texture cache.
 *
 * \par Usage Considerations
 * - Can only be used with primitive types (e.g., \p char, \p int, \p float), with the exception of \p double
 * - Only one TexIteratorRA or TexIteratorRA of a certain \p InputType can be bound at any given time (per host thread)
 *
 * \tparam InputType            The value type of the pointer being wrapped
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p OutputType.  Must have member <tt>OutputType operator()(const InputType &datum)</tt>.
 * \tparam OutputType           The value type of this iterator
 *)
template <typename T>
class TexIteratorRA
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    typedef TexIteratorRA                       self_type;
    typedef T                                   value_type;
    typedef T                                   reference;
    typedef T*                                  pointer;
    typedef std::random_access_iterator_tag     iterator_category;
    typedef int                                 difference_type;

#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Tag identifying iterator type as being texture-bindable
    typedef void TexBindingTag;

private:

    T*                  ptr;
    size_t              tex_align_offset;
    cudaTextureObject_t tex_obj;

public:

    (**
     * \brief Constructor
     *)
    __host__ __device__ __forceinline__ TexIteratorRA()
    :
        ptr(NULL),
        tex_align_offset(0),
        tex_obj(0)
    {}

    /// \brief Bind iterator to texture reference
    cudaError_t BindTexture(
        T               *ptr,                   ///< Native pointer to wrap that is aligned to cudaDeviceProp::textureAlignment
        size_t          bytes,                  ///< Number of items
        size_t          tex_align_offset = 0)   ///< Offset (in items) from ptr denoting the position of the iterator
    {
        this->ptr = ptr;
        this->tex_align_offset = tex_align_offset;

        int ptx_version;
        cudaError_t error = cudaSuccess;
        if (CubDebug(error = PtxVersion(ptx_version))) return error;
        if (ptx_version >= 300)
        {
            // Use texture object
            cudaChannelFormatDesc   channel_desc = cudaCreateChannelDesc<T>();
            cudaResourceDesc        res_desc;
            cudaTextureDesc         tex_desc;
            memset(&res_desc, 0, sizeof(cudaResourceDesc));
            memset(&tex_desc, 0, sizeof(cudaTextureDesc));
            res_desc.resType                = cudaResourceTypeLinear;
            res_desc.res.linear.devPtr      = ptr;
            res_desc.res.linear.desc        = channel_desc;
            res_desc.res.linear.sizeInBytes = bytes;
            tex_desc.readMode               = cudaReadModeElementType;
            return cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
        }
        else
        {
            // Use texture reference
            return TexIteratorRef<T>::BindTexture(ptr);
        }
    }

    /// \brief Unbind iterator to texture reference
    cudaError_t UnbindTexture()
    {
        int ptx_version;
        cudaError_t error = cudaSuccess;
        if (CubDebug(error = PtxVersion(ptx_version))) return error;
        if (ptx_version < 300)
        {
            // Use texture reference
            return TexIteratorRef<T>::UnbindTexture();
        }
        else
        {
            // Use texture object
            return cudaDestroyTextureObject(tex_obj);
        }
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    __host__ __device__ __forceinline__ self_type operator++()
    {
        self_type i = *this;
        ptr++;
        tex_align_offset++;
        return i;
    }

    __host__ __device__ __forceinline__ self_type operator++(int junk)
    {
        ptr++;
        tex_align_offset++;
        return *this;
    }

    __host__ __device__ __forceinline__ reference operator*()
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return *ptr;
#elif (CUB_PTX_ARCH < 300)
        // Use the texture reference
        return tex1Dfetch(TexIteratorRef<T>::ref, tex_align_offset);
#else
        // Use the texture object
        return conversion_op(tex1Dfetch<InputType>(tex_obj, tex_align_offset));
#endif
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator+(SizeT n)
    {
        TexIteratorRA retval;
        retval.ptr = ptr + n;
        retval.tex_align_offset = tex_align_offset + n;
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator-(SizeT n)
    {
        TexIteratorRA retval;
        retval.ptr = ptr - n;
        retval.tex_align_offset = tex_align_offset - n;
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ reference operator[](SizeT n)
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return ptr[n];
#elif (CUB_PTX_ARCH < 300)
        // Use the texture reference
        return tex1Dfetch(TexIteratorRef<T>::ref, tex_align_offset + n);
#else
        // Use the texture object
        return conversion_op(tex1Dfetch<InputType>(tex_obj, tex_align_offset + n));
#endif
    }

    __host__ __device__ __forceinline__ pointer operator->()
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return &(*ptr);
#elif (CUB_PTX_ARCH < 300)
        // Use the texture reference
        return &(tex1Dfetch(TexIteratorRef<T>::ref, tex_align_offset));
#else
        // Use the texture object
        return conversion_op(tex1Dfetch<InputType>(tex_obj, tex_align_offset));
#endif
    }

    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }

    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

};


(**
 * \brief A simple random-access transform iterator for loading primitive values through texture cache and and subsequently applying a transformation operator.
 *
 * \par Overview
 * TexTransformIteratorRA is a random-access iterator that wraps both a native
 * device pointer of type <tt>InputType*</tt> and a unary conversion functor of
 * type \p ConversionOp. \p OutputType references are made by pulling \p InputType
 * values through the texture cache and then transformed them using the
 * \p ConversionOp instance.
 *
 * \par Usage Considerations
 * - Can only be used with primitive types (e.g., \p char, \p int, \p float), with the exception of \p double
 * - Only one TexIteratorRA or TexTransformIteratorRA of a certain \p InputType can be bound at any given time (per host thread)
 *
 * \tparam InputType            The value type of the pointer being wrapped
 * \tparam ConversionOp         Unary functor type for mapping objects of type \p InputType to type \p OutputType.  Must have member <tt>OutputType operator()(const InputType &datum)</tt>.
 * \tparam OutputType           The value type of this iterator
 *)
template <typename OutputType, typename ConversionOp, typename InputType>
class TexTransformIteratorRA
{
public:

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    typedef TexTransformIteratorRA              self_type;
    typedef OutputType                          value_type;
    typedef OutputType                          reference;
    typedef OutputType*                         pointer;
    typedef std::random_access_iterator_tag     iterator_category;
    typedef int                                 difference_type;

#endif  // DOXYGEN_SHOULD_SKIP_THIS

    /// Tag identifying iterator type as being texture-bindable
    typedef void TexBindingTag;

private:

    ConversionOp        conversion_op;
    InputType*          ptr;
    size_t              tex_align_offset;
    cudaTextureObject_t tex_obj;

public:

    (**
     * \brief Constructor
     *)
    TexTransformIteratorRA(
        ConversionOp    conversion_op)          ///< Binary transformation functor
    :
        conversion_op(conversion_op),
        ptr(NULL),
        tex_align_offset(0),
        tex_obj(0)
    {}

    /// \brief Bind iterator to texture reference
    cudaError_t BindTexture(
        InputType*      ptr,                    ///< Native pointer to wrap that is aligned to cudaDeviceProp::textureAlignment
        size_t          bytes,                  ///< Number of items
        size_t          tex_align_offset = 0)   ///< Offset (in items) from ptr denoting the position of the iterator
    {
        this->ptr = ptr;
        this->tex_align_offset = tex_align_offset;

        int ptx_version;
        cudaError_t error = cudaSuccess;
        if (CubDebug(error = PtxVersion(ptx_version))) return error;
        if (ptx_version >= 300)
        {
            // Use texture object
            cudaChannelFormatDesc   channel_desc = cudaCreateChannelDesc<InputType>();
            cudaResourceDesc        res_desc;
            cudaTextureDesc         tex_desc;
            memset(&res_desc, 0, sizeof(cudaResourceDesc));
            memset(&tex_desc, 0, sizeof(cudaTextureDesc));
            res_desc.resType                = cudaResourceTypeLinear;
            res_desc.res.linear.devPtr      = ptr;
            res_desc.res.linear.desc        = channel_desc;
            res_desc.res.linear.sizeInBytes = bytes;
            tex_desc.readMode               = cudaReadModeElementType;
            return cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
        }
        else
        {
            // Use texture reference
            return TexIteratorRef<InputType>::BindTexture(ptr);
        }
    }

    /// \brief Unbind iterator to texture reference
    cudaError_t UnbindTexture()
    {
        int ptx_version;
        cudaError_t error = cudaSuccess;
        if (CubDebug(error = PtxVersion(ptx_version))) return error;
        if (ptx_version >= 300)
        {
            // Use texture object
            return cudaDestroyTextureObject(tex_obj);
        }
        else
        {
            // Use texture reference
            return TexIteratorRef<InputType>::UnbindTexture();
        }
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    __host__ __device__ __forceinline__ self_type operator++()
    {
        self_type i = *this;
        ptr++;
        tex_align_offset++;
        return i;
    }

    __host__ __device__ __forceinline__ self_type operator++(int junk)
    {
        ptr++;
        tex_align_offset++;
        return *this;
    }

    __host__ __device__ __forceinline__ reference operator*()
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return conversion_op(*ptr);
#elif (CUB_PTX_ARCH < 300)
        // Use the texture reference
        return conversion_op(tex1Dfetch(TexIteratorRef<InputType>::ref, tex_align_offset));
#else
        // Use the texture object
        return conversion_op(tex1Dfetch<InputType>(tex_obj, tex_align_offset));
#endif
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator+(SizeT n)
    {
        TexTransformIteratorRA retval(conversion_op);
        retval.ptr = ptr + n;
        retval.tex_align_offset = tex_align_offset + n;
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ self_type operator-(SizeT n)
    {
        TexTransformIteratorRA retval(conversion_op);
        retval.ptr = ptr - n;
        retval.tex_align_offset = tex_align_offset - n;
        return retval;
    }

    template <typename SizeT>
    __host__ __device__ __forceinline__ reference operator[](SizeT n)
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return conversion_op(ptr[n]);
#elif (CUB_PTX_ARCH < 300)
        // Use the texture reference
        return conversion_op(tex1Dfetch(TexIteratorRef<InputType>::ref, tex_align_offset + n));
#else
        // Use the texture object
        return conversion_op(tex1Dfetch<InputType>(tex_obj, tex_align_offset + n));
#endif
    }

    __host__ __device__ __forceinline__ pointer operator->()
    {
#if (CUB_PTX_ARCH == 0)
        // Simply dereference the pointer on the host
        return &conversion_op(*ptr);
#elif (CUB_PTX_ARCH < 300)
        // Use the texture reference
        return &conversion_op(tex1Dfetch(TexIteratorRef<InputType>::ref, tex_align_offset));
#else
        // Use the texture object
        return &conversion_op(tex1Dfetch<InputType>(tex_obj, tex_align_offset));
#endif
    }

    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }

    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

};




(** @} *)       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


(******************************************************************************
 * Common C/C++ macro utilities
 ******************************************************************************)

#pragma once

#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


(**
 * \addtogroup UtilModule
 * @{
 *)

(**
 * Align struct
 *)
#if defined(_WIN32) || defined(_WIN64)
    #define CUB_ALIGN(bytes) __declspec(align(32))
#else
    #define CUB_ALIGN(bytes) __attribute__((aligned(bytes)))
#endif

(**
 * Select maximum(a, b)
 *)
#define CUB_MAX(a, b) (((a) > (b)) ? (a) : (b))

(**
 * Select minimum(a, b)
 *)
#define CUB_MIN(a, b) (((a) < (b)) ? (a) : (b))

(**
 * Quotient of x/y rounded down to nearest integer
 *)
#define CUB_QUOTIENT_FLOOR(x, y) ((x) / (y))

(**
 * Quotient of x/y rounded up to nearest integer
 *)
#define CUB_QUOTIENT_CEILING(x, y) (((x) + (y) - 1) / (y))

(**
 * x rounded up to the nearest multiple of y
 *)
#define CUB_ROUND_UP_NEAREST(x, y) ((((x) + (y) - 1) / (y)) * y)

(**
 * x rounded down to the nearest multiple of y
 *)
#define CUB_ROUND_DOWN_NEAREST(x, y) (((x) / (y)) * y)

(**
 * Return character string for given type
 *)
#define CUB_TYPE_STRING(type) ""#type

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document
    #define CUB_CAT_(a, b) a ## b
    #define CUB_CAT(a, b) CUB_CAT_(a, b)
#endif // DOXYGEN_SHOULD_SKIP_THIS

(**
 * Static assert
 *)
#define CUB_STATIC_ASSERT(cond, msg) typedef int CUB_CAT(cub_static_assert, __LINE__)[(cond) ? 1 : -1]


(** @} *)       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

(**
 * \file
 * Place-holder for prefixing the cub namespace
 *)

#pragma once

// For example:
//#define CUB_NS_PREFIX namespace thrust{ namespace detail {
//#define CUB_NS_POSTFIX } }

#define CUB_NS_PREFIX
#define CUB_NS_POSTFIX

(**
 * \file
 * PTX intrinsics
 *)


#pragma once

#include "util_type.cuh"
#include "util_arch.cuh"
#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


(**
 * \addtogroup UtilModule
 * @{
 *)


(******************************************************************************
 * PTX helper macros
 ******************************************************************************)

(**
 * Register modifier for pointer-types (for inlining PTX assembly)
 *)
#if defined(_WIN64) || defined(__LP64__)
    #define __CUB_LP64__ 1
    // 64-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "l"
    #define _CUB_ASM_PTR_SIZE_ "u64"
#else
    #define __CUB_LP64__ 0
    // 32-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "r"
    #define _CUB_ASM_PTR_SIZE_ "u32"
#endif


(******************************************************************************
 * Inlined PTX intrinsics
 ******************************************************************************)

(**
 * Shift-right then add.  Returns (x >> shift) + addend.
 *)
__device__ __forceinline__ unsigned int SHR_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
    unsigned int ret;
#if __CUDA_ARCH__ >= 200
    asm("vshr.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
        "=r"(ret) : "r"(x), "r"(shift), "r"(addend));
#else
    ret = (x >> shift) + addend;
#endif
    return ret;
}


(**
 * Shift-left then add.  Returns (x << shift) + addend.
 *)
__device__ __forceinline__ unsigned int SHL_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
    unsigned int ret;
#if __CUDA_ARCH__ >= 200
    asm("vshl.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
        "=r"(ret) : "r"(x), "r"(shift), "r"(addend));
#else
    ret = (x << shift) + addend;
#endif
    return ret;
}


(**
 * Bitfield-extract.
 *)
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits source,
    unsigned int bit_start,
    unsigned int num_bits)
{
    unsigned int bits;
#if __CUDA_ARCH__ >= 200
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
#else
    const unsigned int MASK = (1 << num_bits) - 1;
    bits = (source >> bit_start) & MASK;
#endif
    return bits;
}


(**
 * Bitfield-extract for 64-bit types.
 *)
__device__ __forceinline__ unsigned int BFE(
    unsigned long long source,
    unsigned int bit_start,
    unsigned int num_bits)
{
    const unsigned long long MASK = (1ull << num_bits) - 1;
    return (source >> bit_start) & MASK;
}


(**
 * Bitfield insert.  Inserts the first num_bits of y into x starting at bit_start
 *)
__device__ __forceinline__ void BFI(
    unsigned int &ret,
    unsigned int x,
    unsigned int y,
    unsigned int bit_start,
    unsigned int num_bits)
{
#if __CUDA_ARCH__ >= 200
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(y), "r"(x), "r"(bit_start), "r"(num_bits));
#else
    // TODO
#endif
}


(**
 * Three-operand add
 *)
__device__ __forceinline__ unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
{
#if __CUDA_ARCH__ >= 200
    asm("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(x) : "r"(x), "r"(y), "r"(z));
#else
    x = x + y + z;
#endif
    return x;
}


(**
 * Byte-permute. Pick four arbitrary bytes from two 32-bit registers, and
 * reassemble them into a 32-bit destination register
 *)
__device__ __forceinline__ int PRMT(unsigned int a, unsigned int b, unsigned int index)
{
    int ret;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(index));
    return ret;
}


(**
 * Sync-threads barrier.
 *)
__device__ __forceinline__ void BAR(int count)
{
    asm volatile("bar.sync 1, %0;" : : "r"(count));
}


(**
 * Floating point multiply. (Mantissa LSB rounds towards zero.)
 *)
__device__ __forceinline__ float FMUL_RZ(float a, float b)
{
    float d;
    asm("mul.rz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}


(**
 * Floating point multiply-add. (Mantissa LSB rounds towards zero.)
 *)
__device__ __forceinline__ float FFMA_RZ(float a, float b, float c)
{
    float d;
    asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}


(**
 * Terminates the calling thread
 *)
__device__ __forceinline__ void ThreadExit() {
    asm("exit;");
}    


(**
 * Returns the warp lane ID of the calling thread
 *)
__device__ __forceinline__ unsigned int LaneId()
{
    unsigned int ret;
    asm("mov.u32 %0, %laneid;" : "=r"(ret) );
    return ret;
}


(**
 * Returns the warp ID of the calling thread
 *)
__device__ __forceinline__ unsigned int WarpId()
{
    unsigned int ret;
    asm("mov.u32 %0, %warpid;" : "=r"(ret) );
    return ret;
}

(**
 * Returns the warp lane mask of all lanes less than the calling thread
 *)
__device__ __forceinline__ unsigned int LaneMaskLt()
{
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret) );
    return ret;
}

(**
 * Returns the warp lane mask of all lanes less than or equal to the calling thread
 *)
__device__ __forceinline__ unsigned int LaneMaskLe()
{
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_le;" : "=r"(ret) );
    return ret;
}

(**
 * Returns the warp lane mask of all lanes greater than the calling thread
 *)
__device__ __forceinline__ unsigned int LaneMaskGt()
{
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_gt;" : "=r"(ret) );
    return ret;
}

(**
 * Returns the warp lane mask of all lanes greater than or equal to the calling thread
 *)
__device__ __forceinline__ unsigned int LaneMaskGe()
{
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_ge;" : "=r"(ret) );
    return ret;
}

(**
 * Portable implementation of __all
 *)
__device__ __forceinline__ int WarpAll(int cond)
{
#if CUB_PTX_ARCH < 120

    __shared__ volatile int warp_signals[PtxArchProps::MAX_SM_THREADS / PtxArchProps::WARP_THREADS];

    if (LaneId() == 0)
        warp_signals[WarpId()] = 1;

    if (cond == 0)
        warp_signals[WarpId()] = 0;

    return warp_signals[WarpId()];

#else

    return __all(cond);

#endif
}


(**
 * Portable implementation of __any
 *)
__device__ __forceinline__ int WarpAny(int cond)
{
#if CUB_PTX_ARCH < 120

    __shared__ volatile int warp_signals[PtxArchProps::MAX_SM_THREADS / PtxArchProps::WARP_THREADS];

    if (LaneId() == 0)
        warp_signals[WarpId()] = 0;

    if (cond)
        warp_signals[WarpId()] = 1;

    return warp_signals[WarpId()];

#else

    return __any(cond);

#endif
}


/// Generic shuffle-up
template <typename T>
__device__ __forceinline__ T ShuffleUp(
    T               input,              ///< [in] The value to broadcast
    int             src_offset)         ///< [in] The up-offset of the peer to read from
{
    enum
    {
        SHFL_C = 0,
    };

    typedef typename WordAlignment<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);
    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    #pragma unroll
    for (int WORD = 0; WORD < WORDS; ++WORD)
    {
        unsigned int shuffle_word = input_alias[WORD];
        asm(
            "  shfl.up.b32 %0, %1, %2, %3;"
            : "=r"(shuffle_word) : "r"(shuffle_word), "r"(src_offset), "r"(SHFL_C));
        output_alias[WORD] = (ShuffleWord) shuffle_word;
    }

    return output;
}



(** @} *)       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

(**
 * \file
 * Common type manipulation (metaprogramming) utilities
 *)

#pragma once

#include <iostream>
#include <limits>

#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


(**
 * \addtogroup UtilModule
 * @{
 *)



(******************************************************************************
 * Type equality
 ******************************************************************************)

(**
 * \brief Type selection (<tt>IF ? ThenType : ElseType</tt>)
 *)
template <bool IF, typename ThenType, typename ElseType>
struct If
{
    /// Conditional type result
    typedef ThenType Type;      // true
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <typename ThenType, typename ElseType>
struct If<false, ThenType, ElseType>
{
    typedef ElseType Type;      // false
};

#endif // DOXYGEN_SHOULD_SKIP_THIS


(******************************************************************************
 * Conditional types
 ******************************************************************************)


(**
 * \brief Type equality test
 *)
template <typename A, typename B>
struct Equals
{
    enum {
        VALUE = 0,
        NEGATE = 1
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <typename A>
struct Equals <A, A>
{
    enum {
        VALUE = 1,
        NEGATE = 0
    };
};

#endif // DOXYGEN_SHOULD_SKIP_THIS


(******************************************************************************
 * Marker types
 ******************************************************************************)

(**
 * \brief A simple "NULL" marker type
 *)
struct NullType
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document
    template <typename T>
    __host__ __device__ __forceinline__ NullType& operator =(const T& b) { return *this; }
#endif // DOXYGEN_SHOULD_SKIP_THIS
};


(**
 * \brief Allows for the treatment of an integral constant as a type at compile-time (e.g., to achieve static call dispatch based on constant integral values)
 *)
template <int A>
struct Int2Type
{
   enum {VALUE = A};
};


(******************************************************************************
 * Size and alignment
 ******************************************************************************)

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <typename T>
struct WordAlignment
{
    struct Pad
    {
        T       val;
        char    byte;
    };

    enum
    {
        /// The alignment of T in bytes
        ALIGN_BYTES = sizeof(Pad) - sizeof(T)
    };

    /// Biggest shuffle word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename If<(ALIGN_BYTES % 4 == 0),
        int,
        typename If<(ALIGN_BYTES % 2 == 0),
            short,
            char>::Type>::Type                  ShuffleWord;

    /// Biggest volatile word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename If<(ALIGN_BYTES % 8 == 0),
        long long,
        ShuffleWord>::Type                      VolatileWord;

    /// Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename If<(ALIGN_BYTES % 16 == 0),
        longlong2,
        typename If<(ALIGN_BYTES % 8 == 0),
            long long,                                 // needed to get heterogenous PODs to work on all platforms
            ShuffleWord>::Type>::Type           DeviceWord;

    enum
    {
        DEVICE_MULTIPLE = sizeof(DeviceWord) / sizeof(T)
    };

    struct UninitializedBytes
    {
        char buf[sizeof(T)];
    };

    struct UninitializedShuffleWords
    {
        ShuffleWord buf[sizeof(T) / sizeof(ShuffleWord)];
    };

    struct UninitializedVolatileWords
    {
        VolatileWord buf[sizeof(T) / sizeof(VolatileWord)];
    };

    struct UninitializedDeviceWords
    {
        DeviceWord buf[sizeof(T) / sizeof(DeviceWord)];
    };


};


#endif // DOXYGEN_SHOULD_SKIP_THIS


(******************************************************************************
 * Wrapper types
 ******************************************************************************)

(**
 * \brief A storage-backing wrapper that allows types with non-trivial constructors to be aliased in unions
 *)
template <typename T>
struct Uninitialized
{
    /// Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename WordAlignment<T>::DeviceWord DeviceWord;

    enum
    {
        WORDS = sizeof(T) / sizeof(DeviceWord)
    };

    /// Backing storage
    DeviceWord storage[WORDS];

    /// Alias
    __host__ __device__ __forceinline__ T& Alias()
    {
        return reinterpret_cast<T&>(*this);
    }
};


(**
 * \brief A wrapper for passing simple static arrays as kernel parameters
 *)
template <typename T, int COUNT>
struct ArrayWrapper
{
    /// Static array of type \p T
    T array[COUNT];
};


(**
 * \brief Double-buffer storage wrapper for multi-pass stream transformations that require more than one storage array for streaming intermediate results back and forth.
 *
 * Many multi-pass computations require a pair of "ping-pong" storage
 * buffers (e.g., one for reading from and the other for writing to, and then
 * vice-versa for the subsequent pass).  This structure wraps a set of device
 * buffers and a "selector" member to track which is "current".
 *)
template <typename T>
struct DoubleBuffer
{
    /// Pair of device buffer pointers
    T *d_buffers[2];

    ///  Selector into \p d_buffers (i.e., the active/valid buffer)
    int selector;

    /// \brief Constructor
    __host__ __device__ __forceinline__ DoubleBuffer()
    {
        selector = 0;
        d_buffers[0] = NULL;
        d_buffers[1] = NULL;
    }

    /// \brief Constructor
    __host__ __device__ __forceinline__ DoubleBuffer(
        T *d_current,         ///< The currently valid buffer
        T *d_alternate)       ///< Alternate storage buffer of the same size as \p d_current
    {
        selector = 0;
        d_buffers[0] = d_current;
        d_buffers[1] = d_alternate;
    }

    /// \brief Return pointer to the currently valid buffer
    __host__ __device__ __forceinline__ T* Current() { return d_buffers[selector]; }
};



(******************************************************************************
 * Static math
 ******************************************************************************)

(**
 * \brief Statically determine log2(N), rounded up.
 *
 * For example:
 *     Log2<8>::VALUE   // 3
 *     Log2<3>::VALUE   // 2
 *)
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{
    /// Static logarithm value
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };         // Inductive case
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document
template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
    enum {VALUE = (1 << (COUNT - 1) < N) ?                                  // Base case
        COUNT :
        COUNT - 1 };
};
#endif // DOXYGEN_SHOULD_SKIP_THIS


(**
 * \brief Statically determine if N is a power-of-two
 *)
template <int N>
struct PowerOfTwo
{
    enum { VALUE = ((N & (N - 1)) == 0) };
};



(******************************************************************************
 * Pointer vs. iterator detection
 ******************************************************************************)


(**
 * \brief Pointer vs. iterator
 *)
template <typename Tp>
struct IsPointer
{
    enum { VALUE = 0 };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <typename Tp>
struct IsPointer<Tp*>
{
    enum { VALUE = 1 };
};

#endif // DOXYGEN_SHOULD_SKIP_THIS



(******************************************************************************
 * Qualifier detection
 ******************************************************************************)

(**
 * \brief Volatile modifier test
 *)
template <typename Tp>
struct IsVolatile
{
    enum { VALUE = 0 };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <typename Tp>
struct IsVolatile<Tp volatile>
{
    enum { VALUE = 1 };
};

#endif // DOXYGEN_SHOULD_SKIP_THIS


(******************************************************************************
 * Qualifier removal
 ******************************************************************************)

(**
 * \brief Removes \p const and \p volatile qualifiers from type \p Tp.
 *
 * For example:
 *     <tt>typename RemoveQualifiers<volatile int>::Type         // int;</tt>
 *)
template <typename Tp, typename Up = Tp>
struct RemoveQualifiers
{
    /// Type without \p const and \p volatile qualifiers
    typedef Up Type;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <typename Tp, typename Up>
struct RemoveQualifiers<Tp, volatile Up>
{
    typedef Up Type;
};

template <typename Tp, typename Up>
struct RemoveQualifiers<Tp, const Up>
{
    typedef Up Type;
};

template <typename Tp, typename Up>
struct RemoveQualifiers<Tp, const volatile Up>
{
    typedef Up Type;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS



(******************************************************************************
 * Typedef-detection
 ******************************************************************************)


(**
 * \brief Defines a structure \p detector_name that is templated on type \p T.  The \p detector_name struct exposes a constant member \p VALUE indicating whether or not parameter \p T exposes a nested type \p nested_type_name
 *)
#define CUB_DEFINE_DETECT_NESTED_TYPE(detector_name, nested_type_name)  \
    template <typename T>                                               \
    struct detector_name                                                \
    {                                                                   \
        template <typename C>                                           \
        static char& test(typename C::nested_type_name*);               \
        template <typename>                                             \
        static int& test(...);                                          \
        enum                                                            \
        {                                                               \
            VALUE = sizeof(test<T>(0)) < sizeof(int)                    \
        };                                                              \
    };



(******************************************************************************
 * Simple enable-if (similar to Boost)
 ******************************************************************************)

(**
 * \brief Simple enable-if (similar to Boost)
 *)
template <bool Condition, class T = void>
struct EnableIf
{
    /// Enable-if type for SFINAE dummy variables
    typedef T Type;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <class T>
struct EnableIf<false, T> {};

#endif // DOXYGEN_SHOULD_SKIP_THIS


(******************************************************************************
 * Typedef-detection
 ******************************************************************************)

(**
 * \brief Determine whether or not BinaryOp's functor is of the form <tt>bool operator()(const T& a, const T&b)</tt> or <tt>bool operator()(const T& a, const T&b, unsigned int idx)</tt>
 *)
template <typename T, typename BinaryOp>
struct BinaryOpHasIdxParam
{
private:
    template <typename BinaryOpT, bool (BinaryOpT::*)(const T &a, const T &b, unsigned int idx) const>  struct SFINAE1 {};
    template <typename BinaryOpT, bool (BinaryOpT::*)(const T &a, const T &b, unsigned int idx)>        struct SFINAE2 {};
    template <typename BinaryOpT, bool (BinaryOpT::*)(T a, T b, unsigned int idx) const>                struct SFINAE3 {};
    template <typename BinaryOpT, bool (BinaryOpT::*)(T a, T b, unsigned int idx)>                      struct SFINAE4 {};

    template <typename BinaryOpT, bool (BinaryOpT::*)(const T &a, const T &b, int idx) const>           struct SFINAE5 {};
    template <typename BinaryOpT, bool (BinaryOpT::*)(const T &a, const T &b, int idx)>                 struct SFINAE6 {};
    template <typename BinaryOpT, bool (BinaryOpT::*)(T a, T b, int idx) const>                         struct SFINAE7 {};
    template <typename BinaryOpT, bool (BinaryOpT::*)(T a, T b, int idx)>                               struct SFINAE8 {};

    template <typename BinaryOpT> static char Test(SFINAE1<BinaryOpT, &BinaryOpT::operator()> *);
    template <typename BinaryOpT> static char Test(SFINAE2<BinaryOpT, &BinaryOpT::operator()> *);
    template <typename BinaryOpT> static char Test(SFINAE3<BinaryOpT, &BinaryOpT::operator()> *);
    template <typename BinaryOpT> static char Test(SFINAE4<BinaryOpT, &BinaryOpT::operator()> *);

    template <typename BinaryOpT> static char Test(SFINAE5<BinaryOpT, &BinaryOpT::operator()> *);
    template <typename BinaryOpT> static char Test(SFINAE6<BinaryOpT, &BinaryOpT::operator()> *);
    template <typename BinaryOpT> static char Test(SFINAE7<BinaryOpT, &BinaryOpT::operator()> *);
    template <typename BinaryOpT> static char Test(SFINAE8<BinaryOpT, &BinaryOpT::operator()> *);

    template <typename BinaryOpT> static int Test(...);

public:

    /// Whether the functor BinaryOp has a third <tt>unsigned int</tt> index param
    static const bool HAS_PARAM = sizeof(Test<BinaryOp>(NULL)) == sizeof(char);
};



(******************************************************************************
 * Simple type traits utilities.
 *
 * For example:
 *     Traits<int>::CATEGORY             // SIGNED_INTEGER
 *     Traits<NullType>::NULL_TYPE       // true
 *     Traits<uint4>::CATEGORY           // NOT_A_NUMBER
 *     Traits<uint4>::PRIMITIVE;         // false
 *
 ******************************************************************************)

(**
 * \brief Basic type traits categories
 *)
enum Category
{
    NOT_A_NUMBER,
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
    FLOATING_POINT
};


(**
 * \brief Basic type traits
 *)
template <Category _CATEGORY, bool _PRIMITIVE, bool _NULL_TYPE, typename _UnsignedBits>
struct BaseTraits
{
    /// Category
    static const Category CATEGORY      = _CATEGORY;
    enum
    {
        PRIMITIVE       = _PRIMITIVE,
        NULL_TYPE       = _NULL_TYPE,
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

(**
 * Basic type traits (unsigned primitive specialization)
 *)
template <typename _UnsignedBits>
struct BaseTraits<UNSIGNED_INTEGER, true, false, _UnsignedBits>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = UNSIGNED_INTEGER;
    static const UnsignedBits   MIN_KEY     = UnsignedBits(0);
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1);

    enum
    {
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };


    static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key;
    }

    static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key;
    }
};


(**
 * Basic type traits (signed primitive specialization)
 *)
template <typename _UnsignedBits>
struct BaseTraits<SIGNED_INTEGER, true, false, _UnsignedBits>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = SIGNED_INTEGER;
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits   MIN_KEY     = HIGH_BIT;
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1) ^ HIGH_BIT;

    enum
    {
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };

    static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

    static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        return key ^ HIGH_BIT;
    };

};


(**
 * Basic type traits (fp primitive specialization)
 *)
template <typename _UnsignedBits>
struct BaseTraits<FLOATING_POINT, true, false, _UnsignedBits>
{
    typedef _UnsignedBits       UnsignedBits;

    static const Category       CATEGORY    = FLOATING_POINT;
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    static const UnsignedBits   MIN_KEY     = UnsignedBits(-1);
    static const UnsignedBits   MAX_KEY     = UnsignedBits(-1) ^ HIGH_BIT;

    static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
        return key ^ mask;
    };

    static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key)
    {
        UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
        return key ^ mask;
    };

    enum
    {
        PRIMITIVE       = true,
        NULL_TYPE       = false,
    };
};

#endif // DOXYGEN_SHOULD_SKIP_THIS


(**
 * \brief Numeric type traits
 *)
template <typename T> struct NumericTraits :            BaseTraits<NOT_A_NUMBER, false, false, T> {};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <> struct NumericTraits<NullType> :            BaseTraits<NOT_A_NUMBER, false, true, NullType> {};

template <> struct NumericTraits<char> :                BaseTraits<(std::numeric_limits<char>::is_signed) ? SIGNED_INTEGER : UNSIGNED_INTEGER, true, false, unsigned char> {};
template <> struct NumericTraits<signed char> :         BaseTraits<SIGNED_INTEGER, true, false, unsigned char> {};
template <> struct NumericTraits<short> :               BaseTraits<SIGNED_INTEGER, true, false, unsigned short> {};
template <> struct NumericTraits<int> :                 BaseTraits<SIGNED_INTEGER, true, false, unsigned int> {};
template <> struct NumericTraits<long> :                BaseTraits<SIGNED_INTEGER, true, false, unsigned long> {};
template <> struct NumericTraits<long long> :           BaseTraits<SIGNED_INTEGER, true, false, unsigned long long> {};

template <> struct NumericTraits<unsigned char> :       BaseTraits<UNSIGNED_INTEGER, true, false, unsigned char> {};
template <> struct NumericTraits<unsigned short> :      BaseTraits<UNSIGNED_INTEGER, true, false, unsigned short> {};
template <> struct NumericTraits<unsigned int> :        BaseTraits<UNSIGNED_INTEGER, true, false, unsigned int> {};
template <> struct NumericTraits<unsigned long> :       BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long> {};
template <> struct NumericTraits<unsigned long long> :  BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long long> {};

template <> struct NumericTraits<float> :               BaseTraits<FLOATING_POINT, true, false, unsigned int> {};
template <> struct NumericTraits<double> :              BaseTraits<FLOATING_POINT, true, false, unsigned long long> {};

#endif // DOXYGEN_SHOULD_SKIP_THIS


(**
 * \brief Type traits
 *)
template <typename T>
struct Traits : NumericTraits<typename RemoveQualifiers<T>::Type> {};



(** @} *)       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

(**
 * \file
 * Vector type inference utilities
 *)

#pragma once

#include <iostream>

#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


(**
 * \addtogroup UtilModule
 * @{
 *)


(******************************************************************************
 * Vector type inference utilities.  For example:
 *
 * typename VectorHelper<unsigned int, 2>::Type    // Aliases uint2
 *
 ******************************************************************************)

(**
 * \brief Exposes a member typedef \p Type that names the corresponding CUDA vector type if one exists.  Otherwise \p Type refers to the VectorHelper structure itself, which will wrap the corresponding \p x, \p y, etc. vector fields.
 *)
template <typename T, int vec_elements> struct VectorHelper;

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

enum
{
    /// The maximum number of elements in CUDA vector types
    MAX_VEC_ELEMENTS = 4,
};


(**
 * Generic vector-1 type
 *)
template <typename T>
struct VectorHelper<T, 1>
{
    enum { BUILT_IN = false };

    T x;

    typedef VectorHelper<T, 1> Type;
};

(**
 * Generic vector-2 type
 *)
template <typename T>
struct VectorHelper<T, 2>
{
    enum { BUILT_IN = false };

    T x;
    T y;

    typedef VectorHelper<T, 2> Type;
};

(**
 * Generic vector-3 type
 *)
template <typename T>
struct VectorHelper<T, 3>
{
    enum { BUILT_IN = false };

    T x;
    T y;
    T z;

    typedef VectorHelper<T, 3> Type;
};

(**
 * Generic vector-4 type
 *)
template <typename T>
struct VectorHelper<T, 4>
{
    enum { BUILT_IN = false };

    T x;
    T y;
    T z;
    T w;

    typedef VectorHelper<T, 4> Type;
};

(**
 * Macro for expanding partially-specialized built-in vector types
 *)
#define CUB_DEFINE_VECTOR_TYPE(base_type,short_type)                                                            \
  template<> struct VectorHelper<base_type, 1> { typedef short_type##1 Type; enum { BUILT_IN = true }; };         \
  template<> struct VectorHelper<base_type, 2> { typedef short_type##2 Type; enum { BUILT_IN = true }; };         \
  template<> struct VectorHelper<base_type, 3> { typedef short_type##3 Type; enum { BUILT_IN = true }; };         \
  template<> struct VectorHelper<base_type, 4> { typedef short_type##4 Type; enum { BUILT_IN = true }; };

// Expand CUDA vector types for built-in primitives
CUB_DEFINE_VECTOR_TYPE(char,               char)
CUB_DEFINE_VECTOR_TYPE(signed char,        char)
CUB_DEFINE_VECTOR_TYPE(short,              short)
CUB_DEFINE_VECTOR_TYPE(int,                int)
CUB_DEFINE_VECTOR_TYPE(long,               long)
CUB_DEFINE_VECTOR_TYPE(long long,          longlong)
CUB_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
CUB_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
CUB_DEFINE_VECTOR_TYPE(unsigned int,       uint)
CUB_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
CUB_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
CUB_DEFINE_VECTOR_TYPE(float,              float)
CUB_DEFINE_VECTOR_TYPE(double,             double)
CUB_DEFINE_VECTOR_TYPE(bool,               uchar)

// Undefine macros
#undef CUB_DEFINE_VECTOR_TYPE

#endif // DOXYGEN_SHOULD_SKIP_THIS


(** @} *)       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
