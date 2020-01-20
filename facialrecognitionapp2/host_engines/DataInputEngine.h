/**
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-5-19
*/
#ifndef DataInputEngine_ENGINE_H_
#define DataInputEngine_ENGINE_H_
#include "hiaiengine/engine.h"
#include "hiaiengine/data_type.h"
#include "hiaiengine/multitype_queue.h"
#include "BatchImageParaWithScale.h"
#include <iostream>
#include <string>
#include <dirent.h>
#include <memory>
#include <unistd.h>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#define INPUT_SIZE 1
#define OUTPUT_SIZE 1
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_RANDOM_NUMBER 0
#define DEFAULT_DATA_PORT 0

using hiai::Engine;
using namespace std;
using namespace hiai;

//the config of dataset
struct DataConfig{
    std::string path; //the path of dataset
    int batchSize; //batch size of send data
    std::string target; //the target of run
    std::string runMode; //the run mode, run all images or specify images or random images
    int randomNumber; //run random number images
};

//the image info of dataset
struct ImageInfo{
    std::string path; //the path of image
    int format; //the format of image
    int height;
    int width;
    int size;
    int id;
};

class DataInputEngine : public Engine {
public:
    DataInputEngine() :
        data_config_(NULL){}
    ~DataInputEngine(){}
    HIAI_StatusT Init(const hiai::AIConfig& config, const  std::vector<hiai::AIModelDescription>& model_desc) override;
    
    /**
    * @ingroup hiaiengine
    * @brief HIAI_DEFINE_PROCESS : Overloading Engine Process processing logic
    * @[in]: Define an input port, an output port
    */
    HIAI_DEFINE_PROCESS(INPUT_SIZE, OUTPUT_SIZE)
private:
    /**
    * @brief: Send Sentinel Image
    */
    HIAI_StatusT SendSentinelImage();

    /**
    * @brief: get the image buffer
    * @[in]: path, the image path;
    * @[in]: imageBufferPtr, the point of image buffer;
    * @[in]: imageBufferLen, the buffer length;
    * @[in]: frameId, the start of file offset
    * @[return]: bool, if success return true, else return false
    */
    bool GetImageBuffer(const char* path, uint8_t* imageBufferPtr, uint32_t imageBufferLen, uint32_t frameId);

    /**
    * @brief: send batch for Emulator and OI
    * @[in]: batchId, batchId;
    * @[in]: batchNum, the total number of batch;
    * @[in]: imageInfoBatch, the send data;
    * @[return]: HIAI_StatusT
    */
    HIAI_StatusT SendBatch(int batchId, int batchNum, std::shared_ptr<BatchImageParaWithScaleT> imageInfoBatch);

    /**
    * @brief: convert image info to NewImageParaT
    * @[in]: index, index of image in dataset_info_
    * @[out]: imgData, the point of data image
    * @[return]: HIAI_StatusT
    */
    HIAI_StatusT makeImageInfo(NewImageParaT* imgData, int index);

    /**
    * @brief: run images on same side, all engine at same side
    * @[return]: HIAI_StatusT
    */
    HIAI_StatusT RunOnSameSide();

    /**
    * @brief: check whether run images on same side
    * @[return]: HIAI_StatusT
    */
    bool isOnSameSide();

    /**
    * @brief: read data.info file, then convert to dataset_info_
    * @[return]: HIAI_StatusT
    */
    HIAI_StatusT MakeDatasetInfo();

    //the config of dataset
    std::shared_ptr<DataConfig> data_config_;

    //the dataset image infos
    std::vector<ImageInfo> dataset_info_;

    //select image ids, when user select images or random select images
    std::vector<int> select_images_;
};

#endif
