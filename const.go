package onnxruntime

const (
	ORT_LOGGING_LEVEL_VERBOSE ORTLoggingLevel = iota // Verbose informational messages (least severe).
	ORT_LOGGING_LEVEL_INFO                           // Informational messages.
	ORT_LOGGING_LEVEL_WARNING                        // Warning messages.
	ORT_LOGGING_LEVEL_ERROR                          // Error messages.
	ORT_LOGGING_LEVEL_FATAL                          // Fatal error messages (most severe).
)

const (
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED OnnxTensorElementDataType = iota
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
	ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
)

const (
	ORT_DISABLE_ALL     GraphOptimizationLevel = 0
	ORT_ENABLE_BASIC                           = 1
	ORT_ENABLE_EXTENDED                        = 2
	ORT_ENABLE_ALL                             = 99
)
