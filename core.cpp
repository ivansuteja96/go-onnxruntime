#include "core.hpp"

ORTSessionOptions ORTSessionOptions_New() {
    return new Ort::SessionOptions();
}

ORTEnv ORTEnv_New(int logging_level,char* log_env) {
	return new Ort::Env(OrtLoggingLevel(logging_level),log_env);
}

ORTSession* ORTSession_New(ORTEnv ort_env,char* model_location, ORTSessionOptions session_options){  
    auto session = new Ort::Session(*ort_env, model_location, *session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_inputs = session->GetInputCount();
    size_t num_outputs = session->GetOutputCount();

    std::cout << "Num inputs: " << num_inputs << "\n";
    std::cout << "Num outputs: " << num_outputs << "\n";

    std::vector<std::string> input_names = session->GetInputNames();

    for (size_t i = 0; i < input_names.size(); ++i) {
        auto shapes = (*session).GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        printf("Input %ld : name=%s shape=", i, input_names[i].c_str());
        for (size_t i = 0; i < shapes.size(); ++i) {
            printf("%ld", shapes[i]);
            if (i < shapes.size() - 1)
                printf(",");
        }
        printf("\n");
    }

    std::vector<std::string> output_names = session->GetOutputNames();

    for (size_t i = 0; i < output_names.size(); ++i) {
        auto shapes = (*session).GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        printf("Output %ld : name=%s shape=", i, output_names[i].c_str());
        for (size_t i = 0; i < shapes.size(); ++i) {
            printf("%ld", shapes[i]);
            if (i < shapes.size() - 1)
                printf(",");
        }
        printf("\n");
    }

    auto res = new ORTSession{session, input_names, num_inputs, output_names, num_outputs};
    return res;
}

     
void ORTSessionOptions_SetIntraOpNumThreads(ORTSessionOptions session_options, int num_threads) {
    (*session_options).SetIntraOpNumThreads(num_threads);
}

void ORTSessionOptions_SetGraphOptimizationLevel(ORTSessionOptions session_options, int graph_optimization_level) {
    (*session_options).SetGraphOptimizationLevel((GraphOptimizationLevel)2);
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
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<float>(memory_info, (float*)tensor_input.val, tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<uint8_t>(memory_info, (uint8_t*)tensor_input.val, tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<int8_t>(memory_info, (int8_t*)tensor_input.val, tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<uint16_t>(memory_info, (uint16_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<int16_t>(memory_info, (int16_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<int32_t>(memory_info, (int32_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<int64_t>(memory_info, (int64_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<bool>(memory_info, (bool*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<double>(memory_info, (double*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<uint32_t>(memory_info, (uint32_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        (*ort_values).push_back(std::move(Ort::Value::CreateTensor<uint64_t>(memory_info, (uint64_t*)(tensor_input.val), tensor_input.length, (int64_t*)tensor_input.shape.val, (size_t)tensor_input.shape.length)));
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
    Ort::MemoryInfo output_mem_info{"Cpu", OrtDeviceAllocator, 0,
                                  OrtMemTypeDefault};

    Ort::IoBinding binding(*session->session);

    // Bind all inputs by name
    for (size_t i = 0; i < session->num_inputs; ++i) {
        binding.BindInput(session->input_names[i].c_str(),(*ort_values_input)[i]);
    }

    // Bind all outputs dynamically on CPU
    for (size_t i = 0; i < session->num_outputs; ++i) {
        binding.BindOutput(
            session->output_names[i].c_str(),
            output_mem_info
        );
    }

    // ---------- 4. Run ----------
    Ort::RunOptions run_opts;
    (*session->session).Run(run_opts, binding);
    
    std::vector<Ort::Value> outputs = binding.GetOutputValues();

    TensorVector* vector_tv  = (TensorVector*)realloc(vector_tv, outputs.size()*sizeof(*vector_tv));
    for (size_t i = 0; i < outputs.size(); ++i) {
            Ort::Value& out = outputs[i];
            auto output_shape_vector = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
            auto element_type = outputs[i].GetTensorTypeAndShapeInfo().GetElementType();

            int output_length = 1;
            auto output_shape_size = output_shape_vector.size();
            for (int i=0;i<output_shape_size;i++){
                output_length=output_length*output_shape_vector[i];
            }
            auto arr_result = ORTValue_GetTensorMutableData(outputs[i],(size_t)output_length);

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

    TensorVectors tvs = {vector_tv, (int)outputs.size()};
    return tvs;
}

void ORTSession_Free(ORTSession* session) {
	delete session;
}

void TensorVectors_Clear(TensorVectors tvs){
    for (int i = 0; i < tvs.length; i++) {
        free(tvs.arr_vector[i].shape.val);
        free(tvs.arr_vector[i].val);
    }
    free(tvs.arr_vector);
}