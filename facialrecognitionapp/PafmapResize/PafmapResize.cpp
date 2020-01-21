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
#include "PafmapResize.h"
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

#define WSIZE 96
#define HSIZE 72
#define IMAGEW 384
#define IMAGEH 288
#define PAFMAP_SIZE 26

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
HIAI_StatusT PafmapResize::Init(const AIConfig &config,
    const vector<AIModelDescription> &model_desc) {

  HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "robot_dog: ResizeEngine init");
  return HIAI_OK;
}

bool PafmapResize::IsDataHandleWrong(shared_ptr<FaceRecognitionInfo> &face_detail_info) {
  ErrorInfo err_info = face_detail_info->err_info;
  if (err_info.err_code != AppErrorCode::kNone) {
    return false;
  }
  return true;
}

HIAI_StatusT PafmapResize::SendSuccess(
  shared_ptr<FaceRecognitionInfo> &face_recognition_info) {
  // HIAI_StatusT ret = HIAI_OK;
  int scale = IMAGEW/WSIZE;
  OutputT out = face_recognition_info->output_data_vec[1];
  vector<Mat> pafmaps;
    
  shared_ptr<u_int8_t> input_data(new u_int8_t[out.size]);
  memcpy_s(input_data.get(), out.size, out.data.get(), out.size);
  int pafmapSize = 26;
  // if (face_recognition_info->msg == "3"){
  //   pafmapSize = 5;
  // }else{
  //   pafmapSize = 7;
  // }
    
  for(int i = 0; i < pafmapSize; i++){
    Mat v;
    v.create(HSIZE, WSIZE, CV_32FC1);
    v.data = input_data.get() + out.size*i/pafmapSize;
    pafmaps.push_back(v);
  }
    // cout << pafmaps[5].at<float>(3,1) << endl;
  Mat pafMats, resizePafMat;
  merge(pafmaps, pafMats);
  resize(pafMats, resizePafMat, Size(round(scale*WSIZE), round(scale*HSIZE)), 0, 0, INTER_CUBIC);

  int buffer_size = scale*scale*out.size;
  if (face_recognition_info->frame.image_source == 1) {
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "resize: pafmap size after resize is %d", buffer_size);
  }
  
  shared_ptr<u_int8_t> resized_data(new u_int8_t[buffer_size]);
  memcpy_s(resized_data.get(), buffer_size, resizePafMat.data, buffer_size);
  OutputT out1;
  out1.size = buffer_size;
  out1.data = resized_data;
  out1.name = "resized";
  face_recognition_info->output_data_vec[1] = out1;
  // ret = SendData(0, "FaceRecognitionInfo", std::static_pointer_cast<void>(face_recognition_info));

  // HIAI_ENGINE_LOG("VCNN network run success, the total face is %d .",
  //                 face_recognition_info->face_imgs.size());
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

HIAI_IMPL_ENGINE_PROCESS("PafmapResize", PafmapResize, INPUT_SIZE) {
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
