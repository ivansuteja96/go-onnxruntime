package main

import (
	"fmt"
	_ "image/jpeg"
	"log"
	"math/rand"

	"github.com/ivansuteja96/go-onnxruntime"
)

func main() {
	ortEnvDet := onnxruntime.NewORTEnv(onnxruntime.ORT_LOGGING_LEVEL_VERBOSE, "development")
	ortDetSO := onnxruntime.NewORTSessionOptions()

	detModel, err := onnxruntime.NewORTSession(ortEnvDet, "/tmp/model/model.onnx", ortDetSO)
	if err != nil {
		log.Println(err)
		return
	}

	shape := []int64{3, 4, 5}
	input := randFloats(0, 1, int(shape[0]*shape[1]*shape[2]))

	res, err := detModel.Predict([]onnxruntime.TensorValue{
		{
			Value: input,
			Shape: shape,
		},
	})
	if err != nil {
		log.Println(err)
		return
	}

	if len(res) == 0 {
		log.Println("Failed get result")
		return
	}
	fmt.Printf("Success do predict, shape : %+v, result : %+v\n", res[0].Shape, res[0].Value)
}

func randFloats(min, max float32, n int) []float32 {
	res := make([]float32, n)
	for i := range res {
		res[i] = min + rand.Float32()*(max-min)
	}
	return res
}
