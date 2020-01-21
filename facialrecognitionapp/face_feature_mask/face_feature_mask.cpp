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
#include "face_feature_mask.h"
#include "face_feature_train_mean.h"
#include "face_feature_train_std.h"
#include "hiaiengine/log.h"
#include "hiaiengine/data_type_reg.h"
#include "ascenddk/ascend_ezdvpp/dvpp_process.h"
#include <memory>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>

using namespace ascend::utils;
using hiai::Engine;
using namespace std;
using namespace hiai;
using namespace cv;

namespace {
// The image's width need to be resized
const int32_t kResizedImgWidth = 40;

// The image's height need to be resized
const int32_t kResizedImgHeight = 40;

// The rgb image's channel number
const int32_t kRgbChannel = 3;

// For each input, the result should be one tensor
const int32_t kEachResultTensorNum = 10;

// The center's size for the inference result
const float kNormalizedCenterData = 0.5;

const int32_t kSendDataIntervalMiss = 20;
}

/**
* @ingroup hiaiengine
* @brief HIAI_DEFINE_PROCESS : implementaion of the engine
* @[in]: engine name and the number of input
*/
HIAI_StatusT FaceFeatureMaskProcess::Init(const AIConfig &config,
    const vector<AIModelDescription> &model_desc) {
  if (!InitAiModel(config)) {
    return HIAI_ERROR;
  }

  if (!InitNormlizedData()) {
    return HIAI_ERROR;
  }

  return HIAI_OK;
}


bool FaceFeatureMaskProcess::InitAiModel(const AIConfig &config) {
  AIStatus ret = SUCCESS;

  // Define the initialization value for the ai_model_manager_
  if (ai_model_manager_ == nullptr) {
    ai_model_manager_ = make_shared<AIModelManager>();
  }

  vector<AIModelDescription> model_desc_vec;
  AIModelDescription model_desc;

  // Get the model information from the file graph.config
  for (int index = 0; index < config.items_size(); ++index) {
    const AIConfigItem &item = config.items(index);
    if (item.name() == kModelPathParamKey) {
      const char *model_path = item.value().data();
      model_desc.set_path(model_path);
    } else if (item.name() == kBatchSizeParamKey) {
      stringstream ss(item.value());
      ss >> batch_size_;
    } else {
      continue;
    }
  }

  //Invoke the framework's interface to init the information
  model_desc_vec.push_back(model_desc);
  ret = ai_model_manager_->Init(config, model_desc_vec);

  if (ret != SUCCESS) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "AI model init failed!");
    return false;
  }
  return true;
}

bool FaceFeatureMaskProcess::InitNormlizedData() {
  // Load the mean data
  Mat train_mean_value(kResizedImgWidth, kResizedImgHeight, CV_32FC3, (void *)kTrainMean);
  train_mean_ = train_mean_value;
  if (train_mean_.empty()) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "Load mean failed!");
    return false;
  }

  // Load the STD data
  Mat train_std_value(kResizedImgWidth, kResizedImgHeight, CV_32FC3, (void *)kTrainStd);
  train_std_ = train_std_value;
  if (train_std_.empty()) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "Load std failed!");
    return false;
  }
  HIAI_ENGINE_LOG("Load mean and std success!");
  return true;
}

bool FaceFeatureMaskProcess::IsDataHandleWrong(shared_ptr<FaceRecognitionInfo> &face_detail_info) {
  ErrorInfo err_info = face_detail_info->err_info;
  if (err_info.err_code != AppErrorCode::kNone) {
    return false;
  }
  return true;
}

HIAI_StatusT FaceFeatureMaskProcess::SendSuccess(
  shared_ptr<FaceRecognitionInfo> &face_recognition_info) {

  HIAI_ENGINE_LOG("VCNN network run success, the total face is %d .",
                  face_recognition_info->face_imgs.size());
  HIAI_StatusT ret = HIAI_OK;
  do {
    ret = SendData(DEFAULT_DATA_PORT, "FaceRecognitionInfo",
                   static_pointer_cast<void>(face_recognition_info));
    if (ret == HIAI_QUEUE_FULL) {
      HIAI_ENGINE_LOG("Queue is full, sleep 200ms");
      usleep(kSendDataIntervalMiss);
    }
  } while (ret == HIAI_QUEUE_FULL);

  return ret;
}

HIAI_IMPL_ENGINE_PROCESS("face_feature_mask", FaceFeatureMaskProcess, INPUT_SIZE) {
  // args is null, arg0 is image info, arg1 is model info
  if (nullptr == arg0) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "Fail to process invalid message, is null.");
    return HIAI_ERROR;
  }

  // Get the data from last Engine
  // If not correct, Send the message to next node directly
  shared_ptr<FaceRecognitionInfo> face_recognition_info = static_pointer_cast <
      FaceRecognitionInfo > (arg0);
  if (!IsDataHandleWrong(face_recognition_info)) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "The message status is not normal");
    SendData(DEFAULT_DATA_PORT, "FaceRecognitionInfo",
             static_pointer_cast<void>(face_recognition_info));
    return HIAI_ERROR;
  }

  return SendSuccess(face_recognition_info);
}
