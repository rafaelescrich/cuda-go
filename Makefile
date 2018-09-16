APP_NAME=cuda-hello
# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
BINARY_NAME=$(APP_NAME)

all: clean deps test build

build: 
	$(GOBUILD) -o $(BINARY_NAME) -v


test: 
	$(GOTEST) -v ./...

clean: 
	$(GOCLEAN)
	rm -f $(BINARY_NAME)*

run: build
	env $$(grep -v '^#' $(ENV_FILE) | xargs) ./$(BINARY_NAME)

deps:
	$(GOGET) ./...