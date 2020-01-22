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

#include "OpenposeInference.h"

#include <vector>
#include <sstream>

#include "hiaiengine/log.h"
#include "ascenddk/ascend_ezdvpp/dvpp_process.h"

using hiai::Engine;
using hiai::ImageData;
using namespace std;
using namespace ascend::utils;

namespace {

// output port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// model need resized image to 384 * 288
const float kResizeWidth = 384.0;
const float kResizeHeight = 288.0;

// confidence parameter key in graph.confi
const string kConfidenceParamKey = "confidence";

// confidence range (0.0, 1.0]
const float kConfidenceMin = 0.0;
const float kConfidenceMax = 1.0;

// results
// inference output result index
const int32_t kResultIndex = 0;
// each result size (7 float)
const int32_t kEachResultSize = 7;
// attribute index
const int32_t kAttributeIndex = 1;
// score index
const int32_t kScoreIndex = 2;
// left top X-axis coordinate point
const int32_t kLeftTopXaxisIndex = 3;
// left top Y-axis coordinate point
const int32_t kLeftTopYaxisIndex = 4;
// right bottom X-axis coordinate point
const int32_t kRightBottomXaxisIndex = 5;
// right bottom Y-axis coordinate point
const int32_t kRightBottomYaxisIndex = 6;

// face attribute
const float kAttributeFaceLabelValue = 1.0;
const float kAttributeFaceDeviation = 0.00001;

// ratio
const float kMinRatio = 0.0;
const float kMaxRatio = 1.0;

// image source from register
const uint32_t kRegisterSrc = 1;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;
}

// register custom data type
HIAI_REGISTER_DATA_TYPE("FaceRecognitionInfo", FaceRecognitionInfo);
HIAI_REGISTER_DATA_TYPE("FaceRectangle", FaceRectangle);
HIAI_REGISTER_DATA_TYPE("FaceImage", FaceImage);

OpenposeInference::OpenposeInference() {
  ai_model_manager_ = nullptr;
  confidence_ = -1.0;  // initialized as invalid value
}

HIAI_StatusT OpenposeInference::Init(
  const hiai::AIConfig& config,
  const vector<hiai::AIModelDescription>& model_desc) {
  HIAI_ENGINE_LOG("Start initialize!");

  // initialize aiModelManager
  if (ai_model_manager_ == nullptr) {
    ai_model_manager_ = make_shared<hiai::AIModelManager>();
  }

  // get parameters from graph.config
  // set model path to AI model description
  hiai::AIModelDescription fd_model_desc;
  for (int index = 0; index < config.items_size(); index++) {
    const ::hiai::AIConfigItem& item = config.items(index);
    // get model path
    if (item.name() == kModelPathParamKey) {
      const char* model_path = item.value().data();
      fd_model_desc.set_path(model_path);
    } else if (item.name() == kConfidenceParamKey) {  // get confidence
      stringstream ss(item.value());
      ss >> confidence_;
    }
    // else: noting need to do
  }

  // validate confidence
  if (!IsValidConfidence(confidence_)) {
    HIAI_ENGINE_LOG(HIAI_GRAPH_INVALID_VALUE,
                    "confidence invalid, please check your configuration.");
    return HIAI_ERROR;
  }

  // initialize model manager
  vector<hiai::AIModelDescription> model_desc_vec;
  model_desc_vec.push_back(fd_model_desc);
  hiai::AIStatus ret = ai_model_manager_->Init(config, model_desc_vec);
  // initialize AI model manager failed
  if (ret != hiai::SUCCESS) {
    HIAI_ENGINE_LOG(HIAI_GRAPH_INVALID_VALUE, "initialize AI model failed");
    return HIAI_ERROR;
  }

  HIAI_ENGINE_LOG("End initialize!");
  return HIAI_OK;
}

bool OpenposeInference::IsValidConfidence(float confidence) {
  return (confidence > kConfidenceMin) && (confidence <= kConfidenceMax);
}

bool OpenposeInference::IsValidResults(float attr, float score,
                                   const FaceRectangle &rectangle) {
  // attribute is not face (background)
  if (abs(attr - kAttributeFaceLabelValue) > kAttributeFaceDeviation) {
    return false;
  }

  // confidence check
  if ((score < confidence_) || !IsValidConfidence(score)) {
    return false;
  }

  // position check : lt == rb invalid
  if ((rectangle.lt.x == rectangle.rb.x)
      && (rectangle.lt.y == rectangle.rb.y)) {
    return false;
  }
  return true;
}

float OpenposeInference::CorrectionRatio(float ratio) {
  float tmp = (ratio < kMinRatio) ? kMinRatio : ratio;
  return (tmp > kMaxRatio) ? kMaxRatio : tmp;
}

bool OpenposeInference::PreProcess(
  const shared_ptr<FaceRecognitionInfo> &image_handle,
  ImageData<u_int8_t> &resized_image) {
  // input size is less than zero, return failed
  int32_t img_size = image_handle->org_img.size;
  if (img_size <= 0) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "original image size less than or equal zero, size=%d",
                    img_size);
    return false;
  }

  // call ez_dvpp to resize image
  DvppBasicVpcPara resize_para;
  resize_para.input_image_type = image_handle->frame.org_img_format;

  // get original image size and set to resize parameter
  int32_t width = image_handle->org_img.width;
  int32_t height = image_handle->org_img.height;

  // set source resolution ratio
  resize_para.src_resolution.width = width;
  resize_para.src_resolution.height = height;

  // crop parameters, only resize, no need crop, so set original image size
  // set crop left-top point (need even number)
  resize_para.crop_left = 0;
  resize_para.crop_up = 0;
  // set crop right-bottom point (need odd number)
  uint32_t crop_right = ((width >> 1) << 1) - 1;
  uint32_t crop_down = ((height >> 1) << 1) - 1;
  resize_para.crop_right = crop_right;
  resize_para.crop_down = crop_down;

  // set destination resolution ratio (need even number)
  resize_para.dest_resolution.width = kResizeWidth;
  resize_para.dest_resolution.height = kResizeHeight;

  // set input image align or not
  resize_para.is_input_align = image_handle->frame.img_aligned;

  // call
  DvppProcess dvpp_resize_img(resize_para);
  DvppVpcOutput dvpp_output;
  int ret = dvpp_resize_img.DvppBasicVpcProc(image_handle->org_img.data.get(),
                                             img_size, &dvpp_output);
  if (ret != kDvppOperationOk) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "call ez_dvpp failed, failed to resize image.");
    return false;
  }

  // call success, set data and size
  resized_image.data.reset(dvpp_output.buffer, default_delete<u_int8_t[]>());
  resized_image.size = dvpp_output.size;

  if (image_handle->frame.image_source == 1){
    std::ofstream fout("./data2.bin", std::ofstream::binary);
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "dvpp resize size:%d",resized_image.size);
    if(!fout.is_open()){
        printf("bin file open failed");
        exit(0);
    }else{
        fout.write((const char*)(resized_image.data.get()), resized_image.size);
        fout.close();
    }
  }
  return true;
}

bool OpenposeInference::Inference(
  const ImageData<u_int8_t> &resized_image,
  vector<shared_ptr<hiai::IAITensor>> &output_data_vec) {
  // neural buffer
  shared_ptr<hiai::AINeuralNetworkBuffer> neural_buf = shared_ptr <
      hiai::AINeuralNetworkBuffer > (
        new hiai::AINeuralNetworkBuffer(),
        default_delete<hiai::AINeuralNetworkBuffer>());
  neural_buf->SetBuffer((void*) resized_image.data.get(), resized_image.size);

  // input data
  shared_ptr<hiai::IAITensor> input_data = static_pointer_cast<hiai::IAITensor>(neural_buf);
  vector<shared_ptr<hiai::IAITensor>> input_data_vec;
  input_data_vec.push_back(input_data);

  // Call Process
  // 1. create output tensor
  hiai::AIContext ai_context;
  hiai::AIStatus ret = ai_model_manager_->CreateOutputTensor(input_data_vec,
                       output_data_vec);
  // create failed
  if (ret != hiai::SUCCESS) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "call CreateOutputTensor failed");
    return false;
  }

  // 2. process
  HIAI_ENGINE_LOG("aiModelManager->Process start!");
  ret = ai_model_manager_->Process(ai_context, input_data_vec, output_data_vec,
                                   AI_MODEL_PROCESS_TIMEOUT);
  // process failed, also need to send data to post process
  if (ret != hiai::SUCCESS) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "call Process failed");
    return false;
  }
  HIAI_ENGINE_LOG("aiModelManager->Process end!");
  return true;
}

bool OpenposeInference::PostProcess(
  shared_ptr<FaceRecognitionInfo> &image_handle,
  const vector<shared_ptr<hiai::IAITensor>> &output_data_vec) {
  if (image_handle->output_data_vec.empty()) {
    for (unsigned int i = 0; i < output_data_vec.size(); i++) {
      std::shared_ptr<hiai::AISimpleTensor> result_tensor = std::static_pointer_cast<hiai::AISimpleTensor>(output_data_vec[i]);
      // int32_t size = result_tensor->GetSize() / sizeof(float);
      // if(size <= 0){
      //   HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
      //               "the result tensor's size is not correct, size is %d", size);
      //   return false;
      // }
      // float result[size];
      // errno_t mem_ret = memcpy_s(result, sizeof(result), result_tensor->GetBuffer(), result_tensor->GetSize());
      // OutputT out;
      int buffer_size = result_tensor->GetSize();
      // out.name = result_tensor->GetName();
      // out.size = buffer_size;
      // if(out.size <= 0){
      //   HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[OpenPoseInferenceEngine] out.size <= 0");
      //   return HIAI_ERROR;
      // }
      shared_ptr<u_int8_t> input_data(new u_int8_t[buffer_size]);
      memcpy_s(input_data.get(), buffer_size, result_tensor->GetBuffer(), buffer_size);
      OutputT out;
      out.name = "inference result";
      out.size = buffer_size;
      out.data = input_data;
      image_handle->output_data_vec.push_back(out);

      if (image_handle->frame.image_source == 1) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "inference: out size is %d", buffer_size);
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "value is :%f,%f .", *((float*)(input_data.get())), *((float*)(input_data.get()) + 1));

        // int size = buffer_size/sizeof(float);
        // float* result = nullptr;
        // try{
        //   result = new float[size];
        // }catch (const std::bad_alloc& e) {
        //   HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[SaveFilePostProcess_1] alloc output data error!");
        //   return HIAI_ERROR;
        // }
        // int ret  = memcpy_s(result, sizeof(float)*size, out.data.get(), sizeof(float)*size);
        // std::string name(out.name);
        // std::string outFileName = "./" + name + ".txt";
        // // int fd = open(outFileName.c_str(), O_CREAT| O_WRONLY, FIlE_PERMISSION);
        // int oneResultSize = size;
        // for (int k = 0; k < oneResultSize; k++){
        //   // HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "value is :%f",result[k]);
        //   std::string value = std::to_string(result[k]);
        //   if(k > 0){
        //     value = "\n" + value;
        //   }
        //   // ret = write(fd, value.c_str(), value.length());
        // }
        // // ret = close(fd);
        // HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "inference: save finish");
        // delete[] result;
        // result = NULL;
      }

    }
  }
  return true;
}

void OpenposeInference::HandleErrors(
  AppErrorCode err_code, const string &err_msg,
  shared_ptr<FaceRecognitionInfo> &image_handle) {
  // write error log
  HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, err_msg.c_str());

  // set error information
  image_handle->err_info.err_code = err_code;
  image_handle->err_info.err_msg = err_msg;

  // send data
  SendResult(image_handle);
}

void OpenposeInference::SendResult(
  const shared_ptr<FaceRecognitionInfo> &image_handle) {
  
  // when register face, can not discard when queue full
  HIAI_StatusT hiai_ret;
  // for(int i = 0; i < 4; i++){
  //   std::shared_ptr<FaceRecognitionInfo> image_handle1 = std::make_shared<FaceRecognitionInfo>();
  //   image_handle1->frame = image_handle->frame;
  //   image_handle1->err_info = image_handle->err_info;
  //   image_handle1->org_img = image_handle->org_img;
  //   image_handle1->face_imgs = image_handle->face_imgs;
    
  //   OutputT heatmap = image_handle->output_data_vec[0];
  //   image_handle1->output_data_vec.push_back(heatmap);
  //   OutputT pafmap = image_handle->output_data_vec[1];
  //   int buffer_size;
  //   if(i == 3){
  //     buffer_size = pafmap.size*5/26;
  //   }else{
  //     buffer_size = pafmap.size*7/26;
  //   }
  //       // shared_ptr<u_int8_t> send_data(new u_int8_t[buffer_size]);
  //       // memcpy_s(resized_data.get(), buffer_size, resizePafMat.data, buffer_size);
  //   shared_ptr<u_int8_t> send_data(new u_int8_t[buffer_size]);
  //   if(i == 3){
  //     memcpy_s(send_data.get(), buffer_size, pafmap.data.get() + pafmap.size*21/26, buffer_size);
  //           // cout << "send:" << std::to_string(i) << ":"<<*((float*)send_data.get() + 5) << endl;
  //   }else{
  //     memcpy_s(send_data.get(), buffer_size, pafmap.data.get() + i * buffer_size, buffer_size);
  //           // cout << "send:" << std::to_string(i) << ":"<<*((float*)send_data.get() + 7) << endl;
  //   }
  //   OutputT out;
  //   out.size = buffer_size;
  //   out.data = send_data;
  //   out.name = "pafmaps";
  //   image_handle1->output_data_vec.push_back(out);
  //   image_handle1->msg = std::to_string(i);
  //   hiai_ret = SendData(i, "FaceRecognitionInfo", std::static_pointer_cast<void>(image_handle1));
  // }
        
  // if (HIAI_OK != hiai_ret)
  // {
  //   HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[OpenPoseInferenceEngine] SendData failed! error code: %d", hiai_ret);
  // }


  do {
    hiai_ret = SendData(kSendDataPort, "FaceRecognitionInfo",
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

HIAI_StatusT OpenposeInference::Detection(
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

  // resize image
  ImageData<u_int8_t> resized_image;
  if (!PreProcess(image_handle, resized_image)) {
    err_msg = "OpenposeInference call ez_dvpp to resize image failed.";
    HandleErrors(AppErrorCode::kDetection, err_msg, image_handle);
    return HIAI_ERROR;
  }

  // inference
  vector<shared_ptr<hiai::IAITensor>> output_data;
  if (!Inference(resized_image, output_data)) {
    err_msg = "OpenposeInference inference failed.";
    HandleErrors(AppErrorCode::kDetection, err_msg, image_handle);
    return HIAI_ERROR;
  }

  // post process
  if (!PostProcess(image_handle, output_data)) {
    err_msg = "OpenposeInference deal result failed.";
    HandleErrors(AppErrorCode::kDetection, err_msg, image_handle);
    return HIAI_ERROR;
  }

  // send result
  SendResult(image_handle);
  return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("OpenposeInference",
                         OpenposeInference, INPUT_SIZE) {
  HIAI_StatusT ret = HIAI_OK;

  // deal arg0 (camera input)
  if (arg0 != nullptr) {
    HIAI_ENGINE_LOG("camera input will be dealing!");
    shared_ptr<FaceRecognitionInfo> camera_img = static_pointer_cast <
        FaceRecognitionInfo > (arg0);
    ret = Detection(camera_img);
  }

  // deal arg1 (register input)
  if (arg1 != nullptr) {
    HIAI_ENGINE_LOG("register input will be dealing!");
    shared_ptr<FaceRecognitionInfo> register_img = static_pointer_cast <
        FaceRecognitionInfo > (arg1);
    ret = Detection(register_img);
  }
  return ret;
}
