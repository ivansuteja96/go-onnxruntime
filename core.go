package onnxruntime

/*
#include <stdlib.h>
#include "core.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"unsafe"
)

type (
	ORTSessionOptions struct {
		sessOpts C.ORTSessionOptions
	}
	ORTSession struct {
		sess *C.struct_ORTSession
	}
	ORTEnv struct {
		env C.ORTEnv
	}
	ORTValues struct {
		val *C.ORTValues
	}
	ORTLoggingLevel           int
	OnnxTensorElementDataType int
	CudnnConvAlgoSearch       int
	TensorValue               struct {
		Value interface{}
		Shape []int64
	}
	CudaOptions struct {
		DeviceID              int
		CudnnConvAlgoSearch   CudnnConvAlgoSearch
		GPUMemorylimit        int
		ArenaExtendStrategy   bool
		DoCopyInDefaultStream bool
		HasUserComputeStream  bool
	}
)

// NewORTEnv Create onnxruntime environment
func NewORTEnv(loggingLevel ORTLoggingLevel, logEnv string) (ortEnv *ORTEnv) {
	cLogEnv := C.CString(logEnv)
	ortEnv = &ORTEnv{
		env: C.ORTEnv_New(C.int(int(loggingLevel)), cLogEnv),
	}
	C.free(unsafe.Pointer(cLogEnv))
	return ortEnv
}

func (o *ORTEnv) Close() error {
	C.free(unsafe.Pointer(o.env))
	return nil
}

// NewORTSessionOptions return empty onnxruntime session options.
func NewORTSessionOptions() *ORTSessionOptions {
	return &ORTSessionOptions{sessOpts: C.ORTSessionOptions_New()}
}

func (so ORTSessionOptions) Close() error {
	C.free(unsafe.Pointer(so.sessOpts))
	return nil
}

// AppendExecutionProviderCUDA append cuda device to the session options.
func (so ORTSessionOptions) AppendExecutionProviderCUDA(cudaOptions CudaOptions) {
	var intDoCopyInDefaultStream int
	if cudaOptions.DoCopyInDefaultStream {
		intDoCopyInDefaultStream = 1
	}

	var intHasUserComputeStream int
	if cudaOptions.HasUserComputeStream {
		intHasUserComputeStream = 1
	}

	var intArenaExtendStrategy int
	if cudaOptions.ArenaExtendStrategy {
		intArenaExtendStrategy = 1
	}
	C.ORTSessionOptions_AppendExecutionProvider_CUDA(so.sessOpts, C.CudaOptions{
		device_id:                 C.int(cudaOptions.DeviceID),
		cudnn_conv_algo_search:    C.int(cudaOptions.CudnnConvAlgoSearch),
		gpu_mem_limit:             C.int(cudaOptions.GPUMemorylimit),
		arena_extend_strategy:     C.int(intArenaExtendStrategy),
		do_copy_in_default_stream: C.int(intDoCopyInDefaultStream),
		has_user_compute_stream:   C.int(intHasUserComputeStream),
	})
}

// NewORTSession return new onnxruntime session
func NewORTSession(ortEnv *ORTEnv, modelLocation string, sessionOptions *ORTSessionOptions) (ortSession *ORTSession, err error) {
	if ortEnv == nil {
		return ortSession, fmt.Errorf("error nil ort env")
	}
	if _, err = os.Stat(modelLocation); errors.Is(err, os.ErrNotExist) {
		return
	} else if fileExtension := filepath.Ext(modelLocation); fileExtension != ".onnx" {
		err = errors.New("file isn't an onnx model")
		return
	}
	if sessionOptions == nil {
		return ortSession, fmt.Errorf("error nil ort session options")
	}

	cModelLocation := C.CString(modelLocation)
	ortSession = &ORTSession{sess: C.ORTSession_New(ortEnv.env, cModelLocation, sessionOptions.sessOpts)}
	C.free(unsafe.Pointer(cModelLocation))

	return ortSession, nil
}

// newTensorVector generate C.TensorVector
func newTensorVector(tv TensorValue) (ctv C.TensorVector, err error) {
	switch tv.Value.(type) {
	case []float32:
		{
			val, _ := tv.Value.([]float32)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []uint8:
		{
			val := tv.Value.([]uint8)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []int8:
		{
			val := tv.Value.([]int8)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []uint16:
		{
			val := tv.Value.([]uint16)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []int16:
		{
			val := tv.Value.([]int16)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []int32:
		{
			val := tv.Value.([]int32)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []int64:
		{
			val := tv.Value.([]int64)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []bool:
		{
			val := tv.Value.([]bool)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []float64:
		{
			val := tv.Value.([]float64)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []uint32:
		{
			val := tv.Value.([]uint32)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	case []uint64:
		{
			val := tv.Value.([]uint64)
			ctv = C.TensorVector{
				val:       unsafe.Pointer(&val[0]),
				data_type: C.int(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
				length:    C.int(len(val)),
				shape: C.LongVector{
					val:    (*C.long)(&tv.Shape[0]),
					length: C.int(len(tv.Shape)),
				},
			}
		}
	default:
		err = errors.New("invalid data type")
	}
	return
}

// cTensorVectorToGo convert C.TensorVector to Go Value
func cTensorVectorToGo(cVal C.TensorVector) (goVal interface{}, shape []int64, err error) {
	cShapeValue := unsafe.Pointer(cVal.shape.val)
	shape = make([]int64, int64(cVal.shape.length))
	copy(shape, (*[1 << 30]int64)(cShapeValue)[:])

	switch cVal.data_type {
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
		err = errors.New("undefined data type!")
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		{
			tensorValue := make([]float32, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]float32)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		{
			tensorValue := make([]uint8, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]uint8)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		{
			tensorValue := make([]int8, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]int8)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		{
			tensorValue := make([]uint16, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]uint16)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		{
			tensorValue := make([]int16, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]int16)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		{
			tensorValue := make([]int32, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]int32)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		{
			tensorValue := make([]int64, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]int64)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		{
			tensorValue := make([]bool, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]bool)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		{
			tensorValue := make([]float64, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]float64)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		{
			tensorValue := make([]uint32, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]uint32)(cTensorValue)[:])
			goVal = tensorValue
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		{
			tensorValue := make([]uint64, int(cVal.length))
			cTensorValue := unsafe.Pointer(cVal.val)
			copy(tensorValue, (*[1 << 30]uint64)(cTensorValue)[:])
			goVal = tensorValue
		}
	default:
		err = errors.New("invalid data type!")
	}
	return
}

// Predict do prediction from input data
func (ortSession *ORTSession) Predict(inputTensorValues []TensorValue) (result []TensorValue, err error) {
	if ortSession == nil {
		return result, fmt.Errorf("error nil ortSession")
	}

	ortValuesInput := ORTValues{
		val: C.ORTValues_New(),
	}

	for _, inputTensorValue := range inputTensorValues {
		tensorVector, err := newTensorVector(inputTensorValue)
		if err != nil {
			return nil, err
		}
		C.ORTValues_AppendTensor(tensorVector, ortValuesInput.val)
	}

	output := C.ORTSession_Predict(ortSession.sess, ortValuesInput.val)
	outputSize := int(output.length)
	tensorValues := make([]C.TensorVector, outputSize)
	arrVector := unsafe.Pointer(output.arr_vector)
	copy(tensorValues, (*[1 << 30]C.TensorVector)(arrVector)[:])

	result = make([]TensorValue, outputSize)
	for i := 0; i < outputSize; i++ {
		goVal, shape, err := cTensorVectorToGo(tensorValues[i])
		if err != nil {
			return nil, err
		}
		result[i] = TensorValue{
			Value: goVal,
			Shape: shape,
		}
	}
	C.TensorVectors_Clear(output)

	return result, nil
}

func (ortSession *ORTSession) Close() error {
	C.ORTSession_Free(ortSession.sess)
	return nil
}
