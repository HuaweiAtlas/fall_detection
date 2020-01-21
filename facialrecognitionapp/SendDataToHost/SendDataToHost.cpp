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

#include "SendDataToHost.h"

#include <memory>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <string>

#include "hiaiengine/log.h"
#include "hiaiengine/data_type_reg.h"
#include "ascenddk/ascend_ezdvpp/dvpp_process.h"

using namespace std;
using namespace ascend::presenter;
using namespace ascend::utils;
using namespace google::protobuf;

// register custom data type
HIAI_REGISTER_DATA_TYPE("FaceRecognitionInfo", FaceRecognitionInfo);
HIAI_REGISTER_DATA_TYPE("FaceRectangle", FaceRectangle);
HIAI_REGISTER_DATA_TYPE("FaceImage", FaceImage);

HIAI_StatusT SendDataToHost::Init(
    const hiai::AIConfig &config,
    const std::vector<hiai::AIModelDescription> &model_desc) {
  // need do nothing
  return HIAI_OK;
}

HIAI_StatusT SendDataToHost::CheckSendMessageRes(
    const PresenterErrorCode &error_code) {
  if (error_code == PresenterErrorCode::kNone) {
    HIAI_ENGINE_LOG("send message to presenter server successfully.");
    return HIAI_OK;
  }

  HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                  "send message to presenter server failed. error=%d",
                  error_code);
  return HIAI_ERROR;
}

HIAI_StatusT SendDataToHost::SendFeature(
    const shared_ptr<FaceRecognitionInfo> &info) {
  // get channel for send feature (data from camera)
  Channel *channel = PresenterChannels::GetInstance().GetPresenterChannel();
  if (channel == nullptr) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "get channel for send FrameInfo failed.");
    return HIAI_ERROR;
  }

  // front engine deal failed, skip this frame
  if (info->err_info.err_code != AppErrorCode::kNone) {
    HIAI_ENGINE_LOG(
        HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
        "front engine dealing failed, skip this frame, err_code=%d, err_msg=%s",
        info->err_info.err_code, info->err_info.err_msg.c_str());
    return HIAI_ERROR;
  }

  facial_recognition::FrameInfo frame_info;
  frame_info.set_image(
      string(reinterpret_cast<char*>(info->frame.original_jpeg_pic_buffer), info->frame.original_jpeg_pic_size));
  delete info->frame.original_jpeg_pic_buffer;

  // OutputT out = info->output_data_vec[0];
  // OutputT out1 = info->output_data_vec[1];
  // HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "camera: heatmap size is %d", out.size);
  // HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "camera: pafmap size is %d", out1.size);
  // 2. repeated FaceFeature
  vector<FaceImage> face_imgs = info->face_imgs;
  facial_recognition::FaceFeature *feature = nullptr;
  for (int i = 0; i < face_imgs.size(); i++) {
    // every face feature
    feature = frame_info.add_feature();

    // box
    feature->mutable_box()->set_lt_x(face_imgs[i].rectangle.lt.x);
    feature->mutable_box()->set_lt_y(face_imgs[i].rectangle.lt.y);
    feature->mutable_box()->set_rb_x(face_imgs[i].rectangle.rb.x);
    feature->mutable_box()->set_rb_y(face_imgs[i].rectangle.rb.y);

    HIAI_ENGINE_LOG("position is (%d,%d),(%d,%d)",face_imgs[i].rectangle.lt.x,face_imgs[i].rectangle.lt.y,face_imgs[i].rectangle.rb.x,face_imgs[i].rectangle.rb.y);

    // vector
    for (int j = 0; j < face_imgs[i].feature_vector.size(); j++) {
      feature->add_vector(face_imgs[i].feature_vector[j]);
    }
  }

  // send frame information to presenter server
  unique_ptr<Message> resp;
  PresenterErrorCode error_code = channel->SendMessage(frame_info, resp);
  return CheckSendMessageRes(error_code);
}

HIAI_StatusT SendDataToHost::ReplyFeature(
    const shared_ptr<FaceRecognitionInfo> &info) {
  // get channel for reply feature (data from register)
  Channel *channel = PresenterChannels::GetInstance().GetPresenterChannel();
  if (channel == nullptr) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "get channel for send FaceResult failed.");
    return HIAI_ERROR;
  }

  // generate FaceResult
  facial_recognition::FaceResult result;
  result.set_id(info->frame.face_id);
  unique_ptr<Message> resp;

  // 1. front engine dealing failed, send error message
  if (info->err_info.err_code != AppErrorCode::kNone) {
    HIAI_ENGINE_LOG(
        HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
        "front engine dealing failed, reply error response to server");

    result.mutable_response()->set_ret(facial_recognition::kErrorOther);
    result.mutable_response()->set_message(info->err_info.err_msg);

    // send
    PresenterErrorCode error_code = channel->SendMessage(result, resp);
    return CheckSendMessageRes(error_code);
  }

  // 2. dealing success, need set FaceFeature
  result.mutable_response()->set_ret(facial_recognition::kErrorNone);
  // vector<FaceImage> face_imgs = info->face_imgs;
  // int info_size = info->output_data_vec.size();
  OutputT out = info->output_data_vec[0];
  HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "human size is %d", out.size);
  vector<float> humans((float*)(out.data.get()), (float*)(out.data.get()) + out.size/sizeof(float));
  facial_recognition::FaceFeature *face_feature = nullptr;
  for (int i = 0; i < 1; i++) {
    // every face feature
    face_feature = result.add_feature();

    // box
    face_feature->mutable_box()->set_lt_x(0);
    face_feature->mutable_box()->set_lt_y(0);
    face_feature->mutable_box()->set_rb_x(0);
    face_feature->mutable_box()->set_rb_y(0);

    // vector
    for (int j = 0; j < humans.size(); j++) {
      HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "value is %f", humans[j]);
      face_feature->add_vector(humans[j]);
    }
  }

  PresenterErrorCode error_code = channel->SendMessage(result, resp);
  return CheckSendMessageRes(error_code);
}

HIAI_IMPL_ENGINE_PROCESS("SendDataToHost", SendDataToHost, INPUT_SIZE) {
  // HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Senddatatohost start");
  
  // deal arg0 (engine only have one input)
  if (arg0 != nullptr) {
    shared_ptr<FaceRecognitionInfo> image_handle = static_pointer_cast<
        FaceRecognitionInfo>(arg0);

    // deal data from camera
    if (image_handle->frame.image_source == 0) {
      HIAI_ENGINE_LOG("post process dealing data from camera.");
      return SendFeature(image_handle);
    }

    // deal data from register
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "post process dealing data from register.");
    return ReplyFeature(image_handle);
  }
  HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "arg0 is null!");
  return HIAI_ERROR;
}
