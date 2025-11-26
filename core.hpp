#ifdef __cplusplus
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>
using namespace std;

extern "C" {
#endif
#include <onnxruntime_c_api.h>

#ifdef __cplusplus
typedef Ort::SessionOptions* ORTSessionOptions;
    typedef struct ORTSession{
        Ort::Session* session;
        std::vector<std::string> input_names;
        size_t num_inputs;
        std::vector<std::string> output_names;
        size_t num_outputs;
    } ORTSession;
typedef Ort::Env* ORTEnv;
typedef std::vector<Ort::Value> ORTValues;
#else
typedef void* ORTSessionOptions;
typedef struct ORTSession ORTSession;
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

ORTSessionOptions ORTSessionOptions_New();
ORTSession* ORTSession_New(ORTEnv ort_env,char* model_location, ORTSessionOptions session_options);
void ORTSessionOptions_SetIntraOpNumThreads(ORTSessionOptions session_options, int num_threads);
void ORTSessionOptions_SetGraphOptimizationLevel(ORTSessionOptions session_options, int graph_optimization_level);
ORTEnv ORTEnv_New(int logging_level, char* log_env);
TensorVectors ORTSession_Predict(ORTSession* session, ORTValues* ort_values_input);
void ORTSession_Free(ORTSession* session);
ORTValues* ORTValues_New();
void ORTValues_AppendTensor( TensorVector tensor_input, ORTValues* ort_values);
void TensorVectors_Clear(TensorVectors tvs);

#ifdef __cplusplus
}
#endif