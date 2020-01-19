#ifndef ResizeEngine_H
#define ResizeEngine_H

#include "hiaiengine/api.h"
#include "hiaiengine/ai_model_manager.h"
#include "hiaiengine/ai_types.h"
#include "hiaiengine/data_type.h"
#include "hiaiengine/engine.h"
#include "hiaiengine/multitype_queue.h"
#include "hiaiengine/data_type_reg.h"
#include "hiaiengine/ai_tensor.h"
#include "BatchImageParaWithScale.h"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"

#define INPUT_SIZE 1
#define OUTPUT_SIZE 1
using hiai::Engine;

using hiai::Engine;
using namespace std;
using namespace hiai;
using namespace cv;
#define VDEC_MAX_CHANNEL 15; // 0~15 共16个通道

class ResizeEngine : public Engine {
public:
    ~ResizeEngine();

    HIAI_StatusT Init(const AIConfig& config, const std::vector<AIModelDescription>& model_desc);
    HIAI_DEFINE_PROCESS(INPUT_SIZE, OUTPUT_SIZE)

private:
    std::shared_ptr<EngineTransT> tran_data;
};

#endif
