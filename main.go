package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"
	"unsafe"

	cu "github.com/mumax/3/cuda/cu"
)

// Define matrix dimension
const N = 512

func main() {
	// -------------------------------
	// 1) Initialize CUDA
	// -------------------------------
	// In older mumax/3 bindings, Init() returns nothing
	cu.Init(0)

	// Check number of devices
	devCount := cu.DeviceGetCount()
	fmt.Println("Number of CUDA devices:", devCount)

	// Pick device #0
	dev := cu.Device(0)
	fmt.Printf("Using GPU: %s\n", dev.Name())

	// Create a context (older style)
	var ctx cu.CUcontext
	// 0 means default flags; you can also do cu.CtxSchedAuto or so if the API is available
	err := cu.CtxCreate(&ctx, 0, dev)
	if err != cu.SUCCESS {
		log.Fatalf("cu.CtxCreate failed: %v", err)
	}
	defer cu.CtxDestroy(ctx) // ensure cleanup

	// Print some device info
	fmt.Printf("Total GPU Memory: %v\n", dev.TotalMem())

	// -------------------------------
	// 2) Generate random matrices
	// -------------------------------
	A := make([]float32, N*N)
	B := make([]float32, N*N)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < N*N; i++ {
		A[i] = rand.Float32()*2 - 1.0
		B[i] = rand.Float32()*2 - 1.0
	}

	// -------------------------------
	// 3) CPU MatMul
	// -------------------------------
	cpuStart := time.Now()
	Ccpu := matmulCPU(A, B, N)
	cpuTime := time.Since(cpuStart).Milliseconds()
	fmt.Printf("CPU matrix multiplication took: %d ms\n", cpuTime)

	// -------------------------------
	// 4) GPU MatMul
	// -------------------------------
	// 4a) Load PTX module
	mod, errMod := cu.ModuleLoad("matmul.ptx")
	if errMod != cu.SUCCESS {
		log.Fatalf("Failed to load module: %v", errMod)
	}
	defer cu.ModuleUnload(mod)

	// 4b) Get kernel function from module
	kernel, errFunc := cu.ModuleGetFunction(mod, "matMulKernel")
	if errFunc != cu.SUCCESS {
		log.Fatalf("Failed to get kernel function: %v", errFunc)
	}

	// 4c) Allocate device memory
	bytes := int64(N * N * 4) // float32 => 4 bytes
	dA, errA := cu.MemAlloc(bytes)
	if errA != cu.SUCCESS {
		log.Fatalf("Failed to allocate dA: %v", errA)
	}
	defer cu.MemFree(dA)

	dB, errB := cu.MemAlloc(bytes)
	if errB != cu.SUCCESS {
		log.Fatalf("Failed to allocate dB: %v", errB)
	}
	defer cu.MemFree(dB)

	dC, errC := cu.MemAlloc(bytes)
	if errC != cu.SUCCESS {
		log.Fatalf("Failed to allocate dC: %v", errC)
	}
	defer cu.MemFree(dC)

	// 4d) Copy data from host to device
	cu.MemcpyHtoD(dA, unsafe.Pointer(&A[0]), bytes)
	cu.MemcpyHtoD(dB, unsafe.Pointer(&B[0]), bytes)

	// 4e) Launch kernel
	blockSize := 16
	gridSize := (N + blockSize - 1) / blockSize
	sharedMem := 0

	// Must pass int32 pointer for N
	// because kernel signature expects something like (const float*, const float*, float*, int)
	nVal := int32(N)

	// kernel parameters: (A, B, C, N)
	args := []unsafe.Pointer{
		unsafe.Pointer(&dA),
		unsafe.Pointer(&dB),
		unsafe.Pointer(&dC),
		unsafe.Pointer(&nVal),
	}

	gpuStart := time.Now()

	// older LaunchKernel signature is:
	// LaunchKernel(f Function, gx, gy, gz, bx, by, bz, sharedMemBytes int, stream Stream, args []unsafe.Pointer) error
	errK := cu.LaunchKernel(
		kernel,
		gridSize, gridSize, 1, // gx, gy, gz
		blockSize, blockSize, 1, // bx, by, bz
		sharedMem,
		0,    // stream == 0 (default) if you don’t have cu.Stream setup
		args, // your kernel arguments
	)
	if errK != cu.SUCCESS {
		log.Fatalf("Kernel launch failed: %v", errK)
	}

	// 4f) Copy back
	Cgpu := make([]float32, N*N)
	cu.MemcpyDtoH(unsafe.Pointer(&Cgpu[0]), dC, bytes)

	gpuTime := time.Since(gpuStart).Milliseconds()
	fmt.Printf("GPU matrix multiplication took: %d ms\n", gpuTime)

	// -------------------------------
	// 5) Validate results
	// -------------------------------
	if verify(Ccpu, Cgpu, 1e-2) {
		fmt.Println("Verification PASSED: CPU & GPU match within tolerance.")
	} else {
		fmt.Println("Verification FAILED: results differ too much.")
	}
}

// matmulCPU does a basic O(N^3) multiply in Go
func matmulCPU(A, B []float32, n int) []float32 {
	C := make([]float32, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for k := 0; k < n; k++ {
				sum += A[i*n+k] * B[k*n+j]
			}
			C[i*n+j] = sum
		}
	}
	return C
}

// verify checks difference within `tol`
func verify(c1, c2 []float32, tol float32) bool {
	if len(c1) != len(c2) {
		return false
	}
	for i := 0; i < len(c1); i++ {
		diff := c1[i] - c2[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			return false
		}
	}
	return true
}
