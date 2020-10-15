package main

import (
	"fmt"
	"strconv"

	cu "github.com/mumax/3/cuda/cu"
)

func main() {

	cu.Init(0)

	fmt.Println("Hello, I am your GPU:", cu.Device(0).Name())
	fmt.Println("Device's clock rate is: ", cu.Device(0).Properties().ClockRate)
	fmt.Println("Number of devices: " + strconv.Itoa(cu.DeviceGetCount()))
	fmt.Println("Free memory: " + strconv.FormatInt(cu.DeviceGet(0).TotalMem(), 10))

}

// 	var a, b, c [N]int
// 	// fill the arrays 'a' and 'b' on the CPU

//hostIn := make([]float32, N)
// hostIn[0] = 1

// //devIn := cu.MemAlloc(int64(len(hostIn)) * cu.SIZEOF_FLOAT32)
// var devIn cu.DevicePtr
//devIn := cu.MemAlloc(int64(len(hostIn)) * cu.SIZEOF_FLOAT32)
//defer cu.MemFree(devIn)
// cu.MemcpyHtoD(devIn, unsafe.Pointer(&hostIn[0]), devIn.Bytes())

// hostOut := make([]complex64, N/2+1)
// var devOut cu.DevicePtr
// devOut = cu.MemAlloc(int64(len(hostOut)) * cu.SIZEOF_COMPLEX64)
// defer cu.MemFree(devOut)

// plan := cufft.Plan1d(N, cufft.R2C, 1)
// defer plan.Destroy()
// plan.ExecR2C(devIn, devOut)

// cu.MemcpyDtoH(unsafe.Pointer(&hostOut[0]), devOut, devOut.Bytes())

// fmt.Println("hostIn:", hostIn)
// fmt.Println("hostOut:", hostOut)
//}
