#include "core.h"

ORTSessionOptions ORTSessionOptions_New() {
    return new Ort::SessionOptions();
}

void ORTSessionOptions_AppendExecutionProvider_CUDA(ORTSessionOptions session_options, CudaOptions cuda_options) {
	OrtCUDAProviderOptions ort_cuda_options;
    ort_cuda_options.device_id = cuda_options.device_id;
    ort_cuda_options.cudnn_conv_algo_search = (OrtCudnnConvAlgoSearch)cuda_options.cudnn_conv_algo_search;
    ort_cuda_options.gpu_mem_limit = cuda_options.gpu_mem_limit;
    ort_cuda_options.arena_extend_strategy = cuda_options.arena_extend_strategy;
    ort_cuda_options.do_copy_in_default_stream = cuda_options.do_copy_in_default_stream;
    ort_cuda_options.has_user_compute_stream = cuda_options.has_user_compute_stream;
    (*session_options).AppendExecutionProvider_CUDA(ort_cuda_options);
}

ORTEnv ORTEnv_New(int logging_level,char* log_env) {
	return new Ort::Env(OrtLoggingLevel(logging_level),log_env);
}

ORTSession* ORTSession_New(ORTEnv ort_env,char* model_location, ORTSessionOptions session_options){
    auto session = new Ort::Session(*ort_env, model_location, *session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session->GetInputCount();
    char **input_node_names = NULL;
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    input_node_names = (char**)realloc(input_node_names, num_input_nodes*sizeof(*input_node_names));

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        auto shapes = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        input_node_names[i] = inputNodeNameAllocatedStrings.back().get();
        int length = strlen(input_node_names[i]);
        char* varDest = (char*)malloc((length+1) * sizeof(char));
        memcpy(varDest,input_node_names[i], length+1);
        input_node_names[i] = varDest;
        printf("Input : %d, Name : %s, Shape : [", i, input_node_names[i]);
        for (size_t i = 0; i < shapes.size(); ++i) {
            printf("%ld", shapes[i]);
            if (i < shapes.size() - 1){
                printf(",");
            }else{
                printf("]\n");
            }
        }
    }

    size_t num_output_nodes = session->GetOutputCount();
    char **output_node_names = NULL;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    output_node_names = (char**)realloc(output_node_names, num_output_nodes*sizeof(*output_node_names));

    // iterate over all output nodes
    for (int i = 0; i < num_output_nodes; i++) {
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        auto shapes = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        output_node_names[i] = outputNodeNameAllocatedStrings.back().get();
        int length = strlen(output_node_names[i]);
        char* varDest = (char*)malloc((length+1) * sizeof(char));
        memcpy(varDest,output_node_names[i], length+1);
        output_node_names[i] = varDest;
        printf("Output : %d, Name : %s, Shape : [", i, output_node_names[i]);
        for (size_t i = 0; i < shapes.size(); ++i) {
            printf("%ld", shapes[i]);
            if (i < shapes.size() - 1){
                printf(",");
            } else{
                printf("]\n");
            }
        }
    }

    auto res = new ORTSession{session, input_node_names,num_input_nodes, output_node_names, num_output_nodes};
    return res;
}

ORTValues* ORTValues_New(){
    return new ORTValues{};
}

void ORTValues_AppendTensor(TensorVector tensor_input, ORTValues *ort_values){
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
   
    switch (tensor_input.data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        throw std::runtime_error(std::string("undefined data type detected in ORTValues_AppendTensor"));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float*)tensor_input.val, tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<uint8_t>(memory_info, (uint8_t*)tensor_input.val, tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<int8_t>(memory_info, (int8_t*)tensor_input.val, tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<uint16_t>(memory_info, (uint16_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<int16_t>(memory_info, (int16_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, (int32_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, (int64_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<bool>(memory_info, (bool*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<double>(memory_info, (double*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<uint32_t>(memory_info, (uint32_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        (*ort_values).emplace_back(Ort::Value::CreateTensor<uint64_t>(memory_info, (uint64_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length));
    break;
    default: // c++: FLOAT16; onnxruntime: COMPLEX64, COMPLEX128, BFLOAT16; TODO: Implement String method
      throw std::runtime_error(std::string("unsupported data type detected in ORTValues_AppendTensor"));
    }
    return ;
}

void *ORTValue_GetTensorMutableData(Ort::Value& ort_value, size_t size){
    void *res = NULL;
    switch ((ort_value).GetTensorTypeAndShapeInfo().GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        throw std::runtime_error(std::string("undefined data type detected in ORTValue_GetTensorMutableData"));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      res = (void*) malloc(sizeof(float) * size);
      memcpy(res, ort_value.GetTensorMutableData<float>(), sizeof(float) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      res = (void*) malloc(sizeof(uint8_t) * size);
      memcpy(res, ort_value.GetTensorMutableData<uint8_t>(), sizeof(uint8_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      res = (void*) malloc(sizeof(int8_t) * size);
      memcpy(res, ort_value.GetTensorMutableData<int8_t>(), sizeof(int8_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      res = (void*) malloc(sizeof(uint16_t) * size);
      memcpy(res, ort_value.GetTensorMutableData<uint16_t>(), sizeof(uint16_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      res = (void*) malloc(sizeof(int16_t) * size);
      memcpy(res, ort_value.GetTensorMutableData<int16_t>(), sizeof(int16_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      res = (void*) malloc(sizeof(int32_t) * size);
      memcpy(res, ort_value.GetTensorMutableData<int32_t>(), sizeof(int32_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      res = (void*) malloc(sizeof(int64_t) * size);
      memcpy(res, ort_value.GetTensorMutableData<int64_t>(), sizeof(int64_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      res = (void*) malloc(sizeof(bool) * size);
      memcpy(res, ort_value.GetTensorMutableData<bool>(), sizeof(bool) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      res = (void*) malloc(sizeof(double) * size);
      memcpy(res, ort_value.GetTensorMutableData<double>(), sizeof(double) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      res = (void*) malloc(sizeof(uint32_t) * size);
      memcpy(res, ort_value.GetTensorMutableData<uint32_t>(), sizeof(uint32_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      res = (void*) malloc(sizeof(uint64_t) * size);
      memcpy(res, ort_value.GetTensorMutableData<uint64_t>(), sizeof(uint64_t) * size);
    default: // c++: FLOAT16; onnxruntime: COMPLEX64, COMPLEX128, BFLOAT16; TODO: Implement String method
      throw std::runtime_error(std::string("unsupported data type detected in ORTValue_GetTensorMutableData"));
    }
    return res;
}

TensorVectors ORTSession_Predict(ORTSession* session, ORTValues *ort_values_input){
    // score model & input tensor, get back output tensor
    auto output_tensors = (*session->session).Run(Ort::RunOptions{nullptr}, session->input_node_names, (*ort_values_input).data(), session->input_node_names_length, session->output_node_names, session->output_node_names_length);
    
    auto output_tensors_count = output_tensors.size();
    TensorVector* vector_tv  = (TensorVector*)realloc(vector_tv, output_tensors_count*sizeof(*vector_tv));
    for (size_t i = 0; i< output_tensors_count;i++){
        auto output_shape_vector = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        auto element_type = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementType();

        int output_length = 1;
        auto output_shape_size = output_shape_vector.size();
        for (int i=0;i<output_shape_size;i++){
            output_length=output_length*output_shape_vector[i];
        }
        auto arr_result = ORTValue_GetTensorMutableData(output_tensors[i],(size_t)output_length);
        
        auto temp_output_shape_val =  output_shape_vector.data();
        long *output_shape_val = NULL;
        output_shape_val = (long*)realloc(output_shape_val, output_shape_size*sizeof(*output_shape_val));
        for (int i = 0; i < output_shape_size; i++) {
            output_shape_val[i] = (long)temp_output_shape_val[i];
        }
        LongVector output_shape = {output_shape_val, (int)output_shape_size};
        TensorVector fv = {arr_result,element_type, output_shape, output_length};
        vector_tv[i]=fv;
    }

    TensorVectors tvs = {vector_tv, (int)output_tensors_count};
    return tvs;
}

void ORTSession_Free(ORTSession* session) {
	free(session->input_node_names);
	free(session->output_node_names);
	free(session);
}

void TensorVectors_Clear(TensorVectors tvs){
    for (int i = 0; i < tvs.length; i++) {
        free(tvs.arr_vector[i].shape.val);
        free(tvs.arr_vector[i].val);
    }
    free(tvs.arr_vector);
}