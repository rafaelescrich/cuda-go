package main

import (
	"fmt"
	"math/rand"
	"time"
	"unsafe"

	cu "github.com/mumax/3/cuda/cu"
)

const N = 512 // Matrix dimension

func main() {
	// 1) Initialize CUDA (no error return)
	cu.Init(0)

	// 2) Choose device
	devCount := cu.DeviceGetCount()
	fmt.Printf("Number of CUDA devices: %d\n", devCount)

	dev := cu.Device(0)
	fmt.Println("Using GPU:", dev.Name())
	fmt.Println("Total GPU Memory:", dev.TotalMem())

	// 3) Create context
	ctx := cu.CtxCreate(0, dev)
	defer ctx.Destroy()

	// 4) Create random NxN matrices on CPU
	A := make([]float32, N*N)
	B := make([]float32, N*N)
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < N*N; i++ {
		A[i] = rand.Float32()*2 - 1.0
		B[i] = rand.Float32()*2 - 1.0
	}

	// 5) CPU multiplication (for comparison & timing)
	cpuStart := time.Now()
	Ccpu := matmulCPU(A, B, N)
	cpuTime := time.Since(cpuStart).Milliseconds()
	fmt.Printf("CPU multiplication took: %d ms\n", cpuTime)

	// 6) Load PTX module & get kernel
	mod := cu.ModuleLoad("matmul.ptx")
	// No `ModuleUnload(mod)`, since it’s not defined in this version of the library.

	kernel := cu.ModuleGetFunction(mod, "matMulKernel")

	// 7) Allocate device memory & copy data
	sizeBytes := int64(N * N * 4) // float32 => 4 bytes
	dA := cu.MemAlloc(sizeBytes)
	defer cu.MemFree(dA)
	dB := cu.MemAlloc(sizeBytes)
	defer cu.MemFree(dB)
	dC := cu.MemAlloc(sizeBytes)
	defer cu.MemFree(dC)

	cu.MemcpyHtoD(dA, unsafe.Pointer(&A[0]), sizeBytes)
	cu.MemcpyHtoD(dB, unsafe.Pointer(&B[0]), sizeBytes)

	// 8) Launch kernel
	blockSize := 16
	gridSize := (N + blockSize - 1) / blockSize

	nVal := int32(N) // match kernel signature
	args := []unsafe.Pointer{
		unsafe.Pointer(&dA),
		unsafe.Pointer(&dB),
		unsafe.Pointer(&dC),
		unsafe.Pointer(&nVal),
	}

	gpuStart := time.Now()
	cu.LaunchKernel(
		kernel,
		gridSize, gridSize, 1, // grid dimensions (X, Y, Z)
		blockSize, blockSize, 1, // block dimensions (X, Y, Z)
		0,            // sharedMemBytes
		cu.Stream(0), // default stream
		args,
	)

	// 9) Copy result back to CPU
	Cgpu := make([]float32, N*N)
	cu.MemcpyDtoH(unsafe.Pointer(&Cgpu[0]), dC, sizeBytes)

	gpuTime := time.Since(gpuStart).Milliseconds()
	fmt.Printf("GPU multiplication took: %d ms\n", gpuTime)

	// 10) Verify correctness
	if verify(Ccpu, Cgpu, 1e-2) {
		fmt.Println("Results match within tolerance.")
	} else {
		fmt.Println("Mismatch in results.")
	}
}

// Simple O(N^3) CPU multiplication
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

// verify checks if results differ more than tol
func verify(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			return false
		}
	}
	return true
}
