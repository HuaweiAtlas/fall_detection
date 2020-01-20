#include "ResizeEngine.h"
#include <hiaiengine/log.h>
#include <hiaiengine/ai_types.h>
#include "hiaiengine/ai_memory.h"
#include <vector>
#include <unistd.h>
#include <thread>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sstream>
#include <fcntl.h>
#define WSIZE 96
#define HSIZE 72
#define IMAGEW 384
#define IMAGEH 288
#define PAFMAP_SIZE 26
HIAI_StatusT ResizeEngine::Init(const AIConfig& config, const std::vector<AIModelDescription>& model_desc)
{
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "robot_dog: ResizeEngine init");
    return HIAI_OK;
}

ResizeEngine::~ResizeEngine()
{
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "robot_dog: ResizeEngine exit !");
}


HIAI_IMPL_ENGINE_PROCESS("ResizeEngine", ResizeEngine, INPUT_SIZE)
{
    HIAI_StatusT ret = HIAI_OK;
    if (nullptr == arg0) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "robot_dog: ResizeEngine arg0 is null.");
        return HIAI_ERROR;
    }

    // 从默认参数中获取输入engine的数据，转换成定义的格式。
    std::shared_ptr<EngineTransT> tran_data = std::static_pointer_cast<EngineTransT>(arg0);
    // cout << "resize:" << tran_data->msg << endl;
    BatchImageParaWithScaleT image_handle = {tran_data->b_info, tran_data->v_img};
    if (isSentinelImage(std::make_shared<BatchImageParaWithScaleT>(image_handle)))
    {   
        HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[ResizeEngine]sentinel Image, process over.");
        HIAI_StatusT hiaiRet = HIAI_OK;
        do{
            hiaiRet = SendData(0, "EngineTransT", std::static_pointer_cast<void>(tran_data));
            if (HIAI_OK != hiaiRet) {
                if (HIAI_ENGINE_NULL_POINTER == hiaiRet || HIAI_HDC_SEND_MSG_ERROR == hiaiRet || HIAI_HDC_SEND_ERROR == hiaiRet
                    || HIAI_GRAPH_SRC_PORT_NOT_EXIST == hiaiRet || HIAI_GRAPH_ENGINE_NOT_EXIST == hiaiRet) {
                    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[ResizeEngine] SendData error[%d], break.", hiaiRet);
                    break;
                }
                HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[ResizeEngine] SendData return value[%d] not OK, sleep 200ms", hiaiRet);
                usleep(SEND_DATA_INTERVAL_MS);
            }
        } while (HIAI_OK != hiaiRet);
        return hiaiRet;
    }

    // cout << "resize:" << tran_data->msg << endl;
    // tran_data->msg = "resized results";

    int scale = IMAGEW/WSIZE;
    OutputT out = tran_data->output_data_vec[1];
    vector<Mat> pafmaps;
    // struct timeval start1;
    // struct timeval stop1;
    // memset(&start1, 0, sizeof(start1));
    // memset(&stop1, 0, sizeof(stop1));
    // gettimeofday(&start1, NULL);
    
    shared_ptr<u_int8_t> input_data(new u_int8_t[out.size]);
    memcpy_s(input_data.get(), out.size, out.data.get(), out.size);
    int pafmapSize;
    if (tran_data->msg == "3"){
        pafmapSize = 5;
        // cout << "resize:" << tran_data->msg << ":"<<*((float*)input_data.get() + 5) << endl;
    }else{
        pafmapSize = 7;
        // cout << "resize:" << tran_data->msg <<":"<<*((float*)input_data.get() + 7) << endl;
    }
    
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
    shared_ptr<u_int8_t> resized_data(new u_int8_t[buffer_size]);
    memcpy_s(resized_data.get(), buffer_size, resizePafMat.data, buffer_size);
    OutputT out1;
    out1.size = buffer_size;
    out1.data = resized_data;
    out1.name = "resized";
    tran_data->output_data_vec[1] = out1;
    // 5行3列4通道
    // if(tran_data->msg == "3"){
    //     cout << *((float*)resized_data.get() + 5*5*384 + 3*5 + 4) << endl;
    // }else{
    //     cout << *((float*)resized_data.get() + 5*7*384 + 3*7 + 4) << endl;
    // }
    // gettimeofday(&stop1, NULL);
    // double time_used1 = (stop1.tv_sec - start1.tv_sec) *1000+(stop1.tv_usec - start1.tv_usec) / 1000.0;
    // cout << "end,resize process time use is:" << time_used1 << endl;

    ret = SendData(0, "EngineTransT", std::static_pointer_cast<void>(tran_data));

    // gettimeofday(&stop1, NULL);
    // double time_used1 = (stop1.tv_sec - start1.tv_sec) *1000+(stop1.tv_usec - start1.tv_usec) / 1000.0;
    // cout << "end,resize process time use is:" << time_used1 << endl;


    if (ret != HIAI_OK)
    {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "robot_dog: ResizeEngine SendData to device fail! ret = %d", ret);
        return ret;
    }
    // cout << "[resizedEngine] end process!" << endl;
    return HIAI_OK;
}
