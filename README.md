# go-onnxruntime
Unofficial Go binding for Onnxruntime C++ API.
This is used to perform onnx model inference in Go.

## Installation

Download and install go-onnxruntime :

```
go get -v github.com/ivansuteja96/go-onnxruntime
```

The binding requires Onnxruntime C++ and Go 1.14++.

### Onnxruntime C++ Library

The Go binding for Onnxruntime C++ API in this repository is built based on Onnxruntime v1.11.0.

To install Onnxruntime C++ on your system, you can go to [onnxruntime](https://github.com/microsoft/onnxruntime/releases/tag/v1.11.0) and download the assets depends on your system (linux/mac/windows).

The Onnxruntime C++ libraries are expected to be under `/usr/local/lib`.

The Onnxruntime C++ header files are expected to be under `/usrl/local/include`.


### Configure Environmental Variables

Configure the linker environmental variables since the Onnxruntime C++ library is under a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file :

Linux (.bashrc) / macOS (.zshrc)
```
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

After that please run either `source ~/.bashrc`(linux) or `source ~/.zshrc`(macOS).

## How to Run

For the quick experience you can run go-onnxruntime using this command :
```
    docker build --platform linux/arm64/v8 -f dockerfile/Dockerfile_ubuntu_arm64_example -t go-onnxruntime .
    docker run --rm -it  go-onnxruntime:latest 
```

**_Note_** : Currently we only provide Dockerfile for ubuntu arm64 architecture.


## Examples

Examples of using the Go Onnxruntime binding to do model inference are under [examples](examples).

## Credits

Some of the logic of conversion is referenced from https://github.com/c3sr/go-onnxruntime.