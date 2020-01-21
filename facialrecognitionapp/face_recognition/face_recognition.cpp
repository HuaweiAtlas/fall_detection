/**
 * ============================================================================
 *
 * Copyright (C) 2018, Hisilicon Technologies Co., Ltd. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#include "face_recognition.h"

#include <cstdint>
#include <memory>
#include <sstream>

#include "hiaiengine/log.h"
#include "ascenddk/ascend_ezdvpp/dvpp_process.h"

using hiai::Engine;
using hiai::ImageData;
using cv::Mat;
using namespace std;
using namespace ascend::utils;

namespace {
// output port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// level for call Dvpp
const int32_t kDvppToJpegLevel = 100;

// model need resized image to 300 * 300
const float kResizeWidth = 96.0;
const float kResizeHeight = 112.0;

// image source from register
const uint32_t kRegisterSrc = 1;

// The memory size of the BGR image is 3 times that of width*height.
const int32_t kBgrBufferMultiple = 3;

// destination points for aligned face
const float kLeftEyeX = 30.2946;
const float kLeftEyeY = 51.6963;
const float kRightEyeX = 65.5318;
const float kRightEyeY = 51.5014;
const float kNoseX = 48.0252;
const float kNoseY = 71.7366;
const float kLeftMouthCornerX = 33.5493;
const float kLeftMouthCornerY = 92.3655;
const float kRightMouthCornerX = 62.7299;
const float kRightMouthCornerY = 92.2041;

// wapAffine estimate check cols(=2) and rows(=3)
const int32_t kEstimateRows = 2;
const int32_t kEstimateCols = 3;

// flip face
// Horizontally flip for OpenCV
const int32_t kHorizontallyFlip = 1;
// Vertically and Horizontally flip for OpenCV
const int32_t kVerticallyAndHorizontallyFlip = -1;

// inference batch
const int32_t kBatchSize = 4;
// every batch has one aligned face and one flip face, total 2
const int32_t kBatchImgCount = 2;
// inference result index
const int32_t kInferenceResIndex = 0;
// every face inference result should contains 1024 vector (float)
const int32_t kEveryFaceResVecCount = 1024;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;
}

// register custom data type
HIAI_REGISTER_DATA_TYPE("FaceRecognitionInfo", FaceRecognitionInfo);
HIAI_REGISTER_DATA_TYPE("FaceRectangle", FaceRectangle);
HIAI_REGISTER_DATA_TYPE("FaceImage", FaceImage);

FaceRecognition::FaceRecognition() {
  ai_model_manager_ = nullptr;
}

HIAI_StatusT FaceRecognition::Init(
    const hiai::AIConfig& config,
    const vector<hiai::AIModelDescription>& model_desc) {
  HIAI_ENGINE_LOG("Start initialize!");

  // initialize aiModelManager
  if (ai_model_manager_ == nullptr) {
    ai_model_manager_ = make_shared<hiai::AIModelManager>();
  }

  // get parameters from graph.config
  // set model path to AI model description
  hiai::AIModelDescription fg_model_desc;
  for (int index = 0; index < config.items_size(); ++index) {
    const ::hiai::AIConfigItem& item = config.items(index);
    // get model_path
    if (item.name() == kModelPathParamKey) {
      const char* model_path = item.value().data();
      fg_model_desc.set_path(model_path);
    }
    // else: noting need to do
  }

  // initialize model manager
  vector<hiai::AIModelDescription> model_desc_vec;
  model_desc_vec.push_back(fg_model_desc);
  hiai::AIStatus ret = ai_model_manager_->Init(config, model_desc_vec);
  // initialize AI model manager failed
  if (ret != hiai::SUCCESS) {
    HIAI_ENGINE_LOG(HIAI_GRAPH_INVALID_VALUE, "initialize AI model failed");
    return HIAI_ERROR;
  }

  HIAI_ENGINE_LOG("End initialize!");
  return HIAI_OK;
}



void FaceRecognition::SendResult(
    const shared_ptr<FaceRecognitionInfo> &image_handle) {
  HIAI_StatusT hiai_ret;
  // if(!GetOriginPic(image_handle)){
  //   image_handle->err_info.err_code = AppErrorCode::kRecognition;
  //   image_handle->err_info.err_msg = "Get the original pic failed";

  //   HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
  //                   "Engine handle filed, err_code=%d, err_msg=%s",
  //                   image_handle->err_info.err_code,
  //                   image_handle->err_info.err_msg.c_str());
  // }
  // when register face, can not discard when queue full
  do {
    hiai_ret = SendData(0, "FaceRecognitionInfo",
                        static_pointer_cast<void>(image_handle));
    // when queue full, sleep
    if (hiai_ret == HIAI_QUEUE_FULL) {
      HIAI_ENGINE_LOG("queue full, sleep 200ms");
      usleep(kSleepInterval);
    }
  } while (hiai_ret == HIAI_QUEUE_FULL
      && image_handle->frame.image_source == kRegisterSrc);

  // send failed
  if (hiai_ret != HIAI_OK) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "call SendData failed, err_code=%d", hiai_ret);
  }
}

HIAI_StatusT FaceRecognition::Recognition(
    shared_ptr<FaceRecognitionInfo> &image_handle) {
  string err_msg = "";
  if (image_handle->err_info.err_code != AppErrorCode::kNone) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "front engine dealing failed, err_code=%d, err_msg=%s",
                    image_handle->err_info.err_code,
                    image_handle->err_info.err_msg.c_str());
    SendResult(image_handle);
    return HIAI_ERROR;
  }

  // pre-process
  // vector<AlignedFace> aligned_imgs;
  // PreProcess(image_handle->face_imgs, aligned_imgs);

  // need to inference or not
  // if (aligned_imgs.empty()) {
  //   HIAI_ENGINE_LOG("no need to inference any image.");
  //   SendResult(image_handle);
  //   return HIAI_OK;
  // }

  // inference and set results
  // InferenceFeatureVector(aligned_imgs, image_handle->face_imgs);

  // send result
  SendResult(image_handle);
  return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("face_recognition",
    FaceRecognition, INPUT_SIZE) {
  HIAI_StatusT ret = HIAI_OK;

  // deal arg0 (engine only have one input)
  if (arg0 != nullptr) {
    HIAI_ENGINE_LOG("begin to deal face_recognition!");
    shared_ptr<FaceRecognitionInfo> image_handle = static_pointer_cast<
        FaceRecognitionInfo>(arg0);
    ret = Recognition(image_handle);
  }
  return ret;
}
