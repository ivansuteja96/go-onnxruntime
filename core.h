#ifdef __cplusplus
#include <onnxruntime_cxx_api.h>
#include <vector>
using namespace std;

extern "C" {
#endif
#include <onnxruntime_c_api.h>

#ifdef __cplusplus
typedef Ort::SessionOptions* ORTSessionOptions;
typedef struct ORTSession{
    Ort::Session* session;
    char** input_node_names;
    size_t input_node_names_length;
    char** output_node_names;
    size_t output_node_names_length;
} ORTSession;
typedef Ort::Env* ORTEnv;
typedef std::vector<Ort::Value> ORTValues;
#else
typedef void* ORTSessionOptions;
typedef struct ORTSession{
    void* session;
    char** input_node_names;
    char** output_node_names;
    size_t input_node_names_length;
    size_t output_node_names_length;
} ORTSession;
typedef void* ORTEnv;
typedef void* ORTMemoryInfo;
typedef void* ORTValues;
#endif
typedef struct LongVector{
    long* val;
    int length;
} LongVector;
typedef struct TensorVector{
    void* val;
    int data_type;
    LongVector shape;
    int length;
} TensorVector;
typedef struct TensorVectors{
    TensorVector* arr_vector;
    int length;
} TensorVectors;
typedef struct CudaOptions{
    int device_id;
    int cudnn_conv_algo_search;
    int gpu_mem_limit;
    int arena_extend_strategy;
    int do_copy_in_default_stream;
    int has_user_compute_stream;
} CudaOptions;

ORTSessionOptions ORTSessionOptions_New();
ORTSession ORTSession_New(ORTEnv ort_env,char* model_location, ORTSessionOptions session_options);
void ORTSessionOptions_AppendExecutionProvider_CUDA(ORTSessionOptions session_options, CudaOptions cuda_options);
ORTEnv ORTEnv_New(int logging_level, char* log_env);
TensorVectors ORTSession_Predict(ORTSession session, ORTValues* ort_values_input);
ORTValues* ORTValues_New();
void ORTValues_AppendTensor( TensorVector tensor_input, ORTValues* ort_values);
void TensorVectors_Clear(TensorVectors tvs);

#ifdef __cplusplus
}
#endif