# Cuda Go

## Go Sample program (to check Cuda bindings)

```go
package main
import(
  cu "github.com/rafaelescrich/3/cuda"
  "fmt"
  "C"
  "strconv"
)

func main(){
  cu.Init(0);
  fmt.Println("Hello, I am you GPU:", cu.Device(0).Name())
  fmt.Println("Number of devices: " + strconv.Itoa(cu.DeviceGetCount()))
  fmt.Println("Free memory: " + strconv.FormatInt(cu.DeviceGet(0).TotalMem(),10))
}
```

Response message:

```bash
$ ./cuda-go
ello, I am your GPU: GeForce 940MX
Devices clock rate is:  1189000
Number of devices: 1
Free memory: 4242604032
```

## Go Bindings from Mumax Cu

- https://godoc.org/github.com/mumax/3/cuda/cu

## Go Dependencies

- go get github.com/rafaelescrich/3

## Server preinstall

Install C++ And Nvidia dependencies

```text
$ sudo apt-get update
$ sudo apt-get install \
    freeglut3-dev \
    g++-4.9 \
    gcc-4.9 \
    libglu1-mesa-dev \
    libx11-dev \
    libxi-dev \
    libxmu-dev \
    nvidia-modprobe \
    bison \
    flex
```

Install Nvidia latest drivers

```bash
sudo apt-get purge nvidia-*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-367
```

Install last CUDA Platform for Ubuntu

```bash
curl -O https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

Prepare CUDA environment

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Check the CUDA version

```bash
$ nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Sun_Sep__4_22:14:01_CDT_2016
Cuda compilation tools, release 8.0, V8.0.44
```

(Optional) ModernGPU Repository [in order to check CUDA lib]

```bash
git clone https://github.com/moderngpu/moderngpu.git
$ cd moderngpu
$ make
```

Some Cuda program sample

```bash
vi hello.cu
```

```cuda
#include <moderngpu/transform.hxx>

using namespace mgpu;

int main(int argc, char** argv) {
  // The context encapsulates things like an allocator and a stream.
  // By default it prints device info to the console.
  standard_context_t context;

  // Launch five threads to greet us.
  transform([]MGPU_DEVICE(int index) {
    printf("Hello GPU from thread %d\n", index);
  }, 5, context);

  // Synchronize on the context's stream to send the output to the console.
  context.synchronize();

  return 0;
}
```

```bash
nvcc \
      -std=c++11 \
      --expt-extended-lambda \
      -gencode arch=compute_61,code=compute_61 \
      -I ./src/ \
      -o hello \
      hello.cu
```

```bash
$ ./hello

GeForce GTX 1080 : 1835.000 Mhz   (Ordinal 0)
20 SMs enabled. Compute Capability sm_61
FreeMem:   6678MB   TotalMem:   8110MB   64-bit pointers.
Mem Clock: 5005.000 Mhz x 256 bits   (320.3 GB/s)
ECC Disabled


Hello GPU from thread 0
Hello GPU from thread 1
Hello GPU from thread 2
Hello GPU from thread 3
Hello GPU from thread 4
```