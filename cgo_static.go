package onnxruntime

// Changes here should be mirrored in contrib/cgo_static.go and cuda/cgo_static.go.

/*
#cgo CXXFLAGS:   --std=c++17
#cgo !windows CPPFLAGS: -I/usr/local/include
#cgo !windows LDFLAGS: -L/usr/local/lib -lonnxruntime -lstdc++
*/
import "C"
