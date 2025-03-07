APP_NAME       = cuda-go
PTX_SRC        = matmul.cu
PTX_FILE       = matmul.ptx

GOCMD          = go
GOBUILD        = $(GOCMD) build
GOCLEAN        = $(GOCMD) clean
GOTEST         = $(GOCMD) test
GOGET          = $(GOCMD) get
BINARY_NAME    = $(APP_NAME)

# Default target
all: clean deps test ptx build

# Compile CUDA source to PTX
ptx:
	nvcc -arch=sm_61 --ptx $(PTX_SRC) -o $(PTX_FILE)

# Build the Go binary
build:
	$(GOBUILD) -o $(BINARY_NAME) -v

# Run tests
test:
	$(GOTEST) -v ./...

# Clean up
clean:
	$(GOCLEAN)
	rm -f $(BINARY_NAME) $(PTX_FILE)

# Fetch dependencies
deps:
	$(GOGET) ./...

# Local run (assuming you have an ENV_FILE or you don't need env)
run: build
	./$(BINARY_NAME)
