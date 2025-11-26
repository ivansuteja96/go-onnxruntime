package onnxruntime

// Changes here should be mirrored in contrib/cgo_static.go and cuda/cgo_static.go.

/*
#cgo CXXFLAGS: -std=c++17 -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -lonnxruntime -lstdc++
*/
import "C"
