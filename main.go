package main

import (
	"fmt"
	"strconv"

	"github.com/mumax/3/cuda/cu"
)

func main() {
	// cu.Init(0)
	// // fmt.Println("Hello, I am your GPU:", cu.Device(0).Name())
	// // fmt.Println("Device's clock rate is: ", cu.Device(0).Properties().ClockRate)
	// // fmt.Println("Number of devices: " + strconv.Itoa(cu.DeviceGetCount()))
	// // fmt.Println("Free memory: " + strconv.FormatInt(cu.DeviceGet(0).TotalMem(), 10))

	// N := 8
	// hostIn := make([]float32, N)
	// hostIn[0] = 1
	// devIn := cu.MemAlloc(int64(len(hostIn)) * cu.SIZEOF_FLOAT32)
	// defer cu.MemFree(devIn)
	// cu.MemcpyHtoD(devIn, unsafe.Pointer(&hostIn[0]), devIn.Bytes())
	// hostOut := make([]complex64, N/2+1)
	// devOut := cu.MemAlloc(int64(len(hostOut)) * cu.SIZEOF_COMPLEX64)
	// //defer cu.MemFree(devOut)
	// plan := cufft.Plan1d(N, cufft.R2C, 1)
	// //defer plan.Destroy()
	// plan.ExecR2C(devIn, devOut)
	// cu.MemcpyDtoH(unsafe.Pointer(&hostOut[0]), devOut, devOut.Bytes())
	// fmt.Println("hostIn:", hostIn)
	// fmt.Println("hostOut:", hostOut)

	cu.Init(0)

	fmt.Println("Hello, I am your GPU:", cu.Device(0).Name())
	fmt.Println("Device's clock rate is: ", cu.Device(0).Properties().ClockRate)
	fmt.Println("Number of devices: " + strconv.Itoa(cu.DeviceGetCount()))
	fmt.Println("Free memory: " + strconv.FormatInt(cu.DeviceGet(0).TotalMem(), 10))

}

// 	var a, b, c [N]int
// 	// fill the arrays 'a' and 'b' on the CPU

// 	for i := 0; i < N; i++ {

// 		a[i] = -i
// 		b[i] = i * i

// 	}

// 	devA := cu.MemAlloc(N)

// 	defer cu.MemFree(devA)

// 	add(&a, &b, &c)

// 	fmt.Println(c)

// }

// func add(a, b, c *[N]int) {
// 	tid := 0 // this is CPU zero, so we start at zero

// 	for tid < N {
// 		c[tid] = a[tid] + b[tid]
// 		tid++ // we have one CPU, so we increment by one
// 	}
// }

// 	N := 3

// 	aslice := [3]int{7, 11, 13}
// 	bslice := [3]int{89, 54, 67}
// 	cslice := [3]int{19, 12, 34}

// 	a := cuda.NewSlice(N, aslice)
// 	b := cuda.NewSlice(N, bslice)
// 	c := cuda.NewSlice(N, cslice)

// 	defer a.Free()
// 	defer b.Free()
// 	defer c.Free()

// 	cuda.MemCpyHtoD([]float32{0, -1, -2})
// 	cuda.MemCpyHtoD([]float32{0, 1, 4})
// 	cfg := Make1DConfig(N)
// 	add_kernel(a.Ptr(), b.Ptr(), c.Ptr(), cfg)

// 	fmt.Println("result:", a.HostCopy())
// }
