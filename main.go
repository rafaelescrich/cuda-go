package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"
	"unsafe"

	cu "github.com/mumax/3/cuda/cu"
)

// Adjust matrix dimension as needed
const N = 512

func main() {
	// ---------------------------
	// Initialize CUDA
	// ---------------------------
	if err := cu.Init(0); err != nil {
		log.Fatalf("Failed to initialize CUDA: %v", err)
	}

	// Print some device info
	devCount := cu.DeviceGetCount()
	fmt.Println("Number of CUDA devices:", devCount)
	device := cu.Device(0)
	ctx, err := device.MakeContext(cu.SchedAuto)
	if err != nil {
		log.Fatalf("Failed to create CUDA context: %v", err)
	}
	defer ctx.Destroy()

	fmt.Printf("Using GPU: %s\n", device.Name())
	fmt.Printf("Device Clock Rate: %d\n", device.Properties().ClockRate)
	fmt.Printf("Total GPU Memory: %d bytes\n", device.TotalMem())

	// ---------------------------
	// Generate random NxN matrices
	// ---------------------------
	A := make([]float32, N*N)
	B := make([]float32, N*N)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < N*N; i++ {
		A[i] = rand.Float32()*2 - 1.0 // random [-1, 1]
		B[i] = rand.Float32()*2 - 1.0 // random [-1, 1]
	}

	// ---------------------------
	// CPU matrix multiplication
	// ---------------------------
	cpuStart := time.Now()
	Ccpu := matmulCPU(A, B, N)
	cpuTime := time.Since(cpuStart).Milliseconds()
	fmt.Printf("CPU matrix multiplication took: %d ms\n", cpuTime)

	// ---------------------------
	// GPU matrix multiplication
	// ---------------------------
	// 1) Load PTX module
	mod := cu.Module{}
	err = mod.Load("matmul.ptx")
	if err != nil {
		log.Fatalf("Failed to load PTX module: %v", err)
	}
	defer mod.Unload()

	// 2) Get kernel function
	kernel, err := mod.Function("matMulKernel")
	if err != nil {
		log.Fatalf("Failed to get kernel function: %v", err)
	}

	// 3) Allocate device memory
	sizeInBytes := int64(N * N * 4) // float32 = 4 bytes
	dA := cu.MemAlloc(sizeInBytes)
	dB := cu.MemAlloc(sizeInBytes)
	dC := cu.MemAlloc(sizeInBytes)
	defer func() {
		cu.MemFree(dA)
		cu.MemFree(dB)
		cu.MemFree(dC)
	}()

	// 4) Copy data from host to device
	cu.MemcpyHtoD(dA, unsafe.Pointer(&A[0]), sizeInBytes)
	cu.MemcpyHtoD(dB, unsafe.Pointer(&B[0]), sizeInBytes)

	// 5) Set up block and grid dimensions
	blockSize := 16
	gridSize := (N + blockSize - 1) / blockSize

	// Prepare kernel arguments
	args := []unsafe.Pointer{
		unsafe.Pointer(&dA),
		unsafe.Pointer(&dB),
		unsafe.Pointer(&dC),
		unsafe.Pointer(&N),
	}

	// Record GPU time
	gpuStart := time.Now()

	// 6) Launch kernel
	err = cu.LaunchKernel(
		kernel,
		int(gridSize), // gridDimX
		int(gridSize), // gridDimY
		1,             // gridDimZ
		blockSize,     // blockDimX
		blockSize,     // blockDimY
		1,             // blockDimZ
		0,             // sharedMemBytes
		cu.Stream{},   // stream
		args,          // kernel arguments
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to launch kernel: %v", err)
	}

	// 7) Copy results back
	Cgpu := make([]float32, N*N)
	cu.MemcpyDtoH(unsafe.Pointer(&Cgpu[0]), dC, sizeInBytes)

	gpuTime := time.Since(gpuStart).Milliseconds()
	fmt.Printf("GPU matrix multiplication took: %d ms\n", gpuTime)

	// ---------------------------
	// (Optional) Verify correctness
	// ---------------------------
	if verify(Ccpu, Cgpu, 1e-2) {
		fmt.Println("Verification PASSED: CPU and GPU results match within tolerance.")
	} else {
		fmt.Println("Verification FAILED: CPU and GPU results differ too much.")
	}

	// Keep the program from exiting if you want to see logs
	// os.Exit(0)
}

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

// verify checks whether the difference between arrays c1 and c2
// is within a certain tolerance.
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
