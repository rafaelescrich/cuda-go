package main

import (
	"C"
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
