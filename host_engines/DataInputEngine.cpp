/**
*
* Copyright(c)<2018>, <Huawei Technologies Co.,Ltd>
*
* @version 1.0
*
* @date 2018-5-19
*/
#include "DataInputEngine.h"
#include "hiaiengine/log.h"
#include "hiaiengine/data_type_reg.h"
#include "hiaiengine/ai_memory.h"
#include <memory>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <map>

const static std::string SIMULATOR_LOCAL = "Simulator_local";
const static std::string OI = "OI";
const int DVPP_BUFFER_ALIGN_SIZE = 128;
extern struct timeval start1;
/**
* @brief: read data.info file, then convert to dataset_info_
* @[return]: HIAI_StatusT
*/
HIAI_StatusT DataInputEngine::MakeDatasetInfo(){
    //get the data info file path
    std::string datainfo_path = data_config_->path;
    while(datainfo_path.back() == '/' || datainfo_path.back() == '\\'){
        datainfo_path.pop_back();
    }
    std::size_t datasetNameIndex = datainfo_path.find_last_of("/\\");
    std::string dataInfoPath = data_config_->path + "." + datainfo_path.substr(datasetNameIndex + 1) + "_data.info";

    //open file
    ifstream fin(dataInfoPath.c_str());
    if (!fin.is_open())
    {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Open %s failed!", dataInfoPath.c_str());
        return HIAI_ERROR;
    }

    std::string line_content;

    //read first line
    std::getline(fin, line_content);
    std::stringstream lineStr1(line_content);
    std::string datasetName;
    int totalFileNum = 0;
    lineStr1 >> datasetName >> totalFileNum;
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "%s total count is %u.", datasetName.c_str(), totalFileNum);

    int format = -2;
    int fileNumByFormat = 0;
    dataset_info_.clear();
    int count = 0;
    while(count < totalFileNum){
        //read format and image count
        std::getline(fin, line_content);
        std::stringstream lineStrFormat(line_content);
        lineStrFormat >> format >> fileNumByFormat;
        for (int i = 0; i < fileNumByFormat; i++) {
            //read each image info
            std::getline(fin, line_content);
            ImageInfo imageInfo;
            imageInfo.format = format;
            std::stringstream lineStr(line_content);
            lineStr >> imageInfo.id >> imageInfo.path >> imageInfo.width >> imageInfo.height >> imageInfo.size;
            dataset_info_.push_back(imageInfo);
        }
        count += fileNumByFormat;
    }
    //close file
    fin.close();
    return HIAI_OK;
}

HIAI_StatusT DataInputEngine::Init(const hiai::AIConfig& config, const  std::vector<hiai::AIModelDescription>& model_desc)
{
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DataInputEngine] start init!");

    data_config_ = std::make_shared<DataConfig>();
    if(data_config_ == NULL){
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] make shared for DataConfig error!");
        return HIAI_ERROR;
    }
    data_config_->batchSize = DEFAULT_BATCH_SIZE;
    data_config_->runMode = "all";

    //read the config of dataset
    for (int index = 0; index < config.items_size(); ++index)
    {
        const ::hiai::AIConfigItem& item = config.items(index);
        std::string name = item.name();
        if(name == "path"){
            data_config_->path = item.value();
        }
        else if(name == "target"){
            data_config_->target = item.value();
        }
    }
    char path[PATH_MAX] = {0};
    if(realpath(data_config_->path.c_str(), path) == NULL){
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] can not find path!");
        return HIAI_ERROR;
    }

    //get the dataset image info
    HIAI_StatusT ret = MakeDatasetInfo();

    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DataInputEngine] end init!");
    return ret;
}

/**
* @brief: get the image buffer
* @[in]: path, the image path;
* @[in]: imageBufferPtr, the point of image buffer;
* @[in]: imageBufferLen, the buffer length;
* @[in]: frameId, the start of file offset
* @[return]: bool, if success return true, else return false
*/
bool DataInputEngine::GetImageBuffer(const char* path, uint8_t* imageBufferPtr, uint32_t imageBufferLen, uint32_t frameId){
    bool ret = false;
    FILE * file = fopen64(path, "r");
    if (NULL == file)
    {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] Error: open file %s failed", path);
        return ret;
    }
    do
    {
        unsigned long imageFseek = ((unsigned  long)frameId)*((unsigned  long)imageBufferLen);
        if (0 != fseeko64(file, imageFseek, SEEK_SET))
        {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] fseeko64 offset = %u failed in GetImageBuffer", frameId * imageBufferLen);
            break;
        }
        if (imageBufferLen != fread(imageBufferPtr, 1, imageBufferLen, file))
        {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] fread length = %u failed in GetImageBuffer", imageBufferLen);
            break;
        }
        ret = true;
    } while (0);

    fclose(file);
    return ret;
}


/**
* @brief free the buffer malloced by HIAI:MALLOC
*/
static void FreeImageBuffer(uint8_t* ptr){
    if (ptr == NULL) {
        return;
    }
    HIAI_StatusT ret = HIAI_OK;
    #if defined(IS_OI)
        ret = hiai::HIAIMemory::HIAI_DVPP_DFree(ptr);
    #else
        ret = hiai::HIAIMemory::HIAI_DFree(ptr);
    #endif
    if (HIAI_OK != ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] DFree buffer error!");
    }
    ptr = NULL;
}

/**
* @brief: convert image info to NewImageParaT
* @[in]: index, index of image in dataset_info_
* @[out]: imgData, the point of data image
* @[return]: HIAI_StatusT
*/
HIAI_StatusT DataInputEngine::makeImageInfo(NewImageParaT* imgData, int index) {
    if(index < 0 || (uint32_t)index >= dataset_info_.size()){
        return HIAI_ERROR;
    }
    imgData->img.format = (IMAGEFORMAT)dataset_info_[index].format;
    imgData->img.width = dataset_info_[index].width ;
    imgData->img.height = dataset_info_[index].height;
    std::string imageFullPath = data_config_->path + dataset_info_[index].path;

    uint8_t * imageBufferPtr = NULL;
    HIAI_StatusT get_ret = HIAI_OK;
    #if defined(IS_OI)
        //run on same side with dvpp
        if((ImageType)dataset_info_[index].format == IMAGE_TYPE_JPEG){
        // transfer jepg to imagepreprocess use dvpp jepgd need to add 8 bit for check
            imgData->img.size = dataset_info_[index].size + 8;
        }else{
            imgData->img.size = dataset_info_[index].size;
        }
        //run on same side with dvpp need to make the mem align to 128(dvpp need)
        int alignBufferSize = (int)ceil(1.0 * imgData->img.size / DVPP_BUFFER_ALIGN_SIZE) * DVPP_BUFFER_ALIGN_SIZE;
        get_ret = hiai::HIAIMemory::HIAI_DVPP_DMalloc(alignBufferSize, (void*&)imageBufferPtr);
    #else
        imgData->img.size = dataset_info_[index].size;
        get_ret = hiai::HIAIMemory::HIAI_DMalloc(imgData->img.size, (void*&)imageBufferPtr);
    #endif

    if(HIAI_OK != get_ret || NULL == imageBufferPtr){
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] DMalloc buffer error!");
        return HIAI_ERROR;
    }
    if(imageBufferPtr == NULL){
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] alloc buffer error in makeImageInfo");
        return HIAI_ERROR;
    }

    bool ret = GetImageBuffer(imageFullPath.c_str(), imageBufferPtr, dataset_info_[index].size, 0);

    if(!ret){
        FreeImageBuffer(imageBufferPtr);
        return HIAI_ERROR;
    }
    // free imageBufferPtr with function FreeImageBuffer()
    shared_ptr<uint8_t> data(imageBufferPtr, FreeImageBuffer);
    imgData->img.data = data;
    return HIAI_OK;
}

/**
* @brief: send batch for Emulator and OI
* @[in]: frameId, frameId;
* @[in]: totalCount, the total number of batch;
* @[in]: imageInfoBatch, the send data;
* @[return]: HIAI_StatusT
*/
HIAI_StatusT DataInputEngine::SendBatch(int frameId, int totalCount, std::shared_ptr<BatchImageParaWithScaleT> imageInfoBatch){
    HIAI_StatusT hiai_ret = HIAI_OK;
    imageInfoBatch->b_info.batch_size = imageInfoBatch->v_img.size();
    imageInfoBatch->b_info.max_batch_size = data_config_->batchSize;
    imageInfoBatch->b_info.batch_ID = frameId;
    imageInfoBatch->b_info.is_first = (frameId == 0 ? true : false);
    imageInfoBatch->b_info.is_last = (frameId == totalCount - 1 ? true : false);

    do{
        hiai_ret = SendData(DEFAULT_DATA_PORT, "BatchImageParaWithScaleT", std::static_pointer_cast<void>(imageInfoBatch));
        if(HIAI_QUEUE_FULL == hiai_ret)
        {
            HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DataInputEngine] queue full, sleep 200ms");
            usleep(SEND_DATA_INTERVAL_MS);
        }
    }while(hiai_ret == HIAI_QUEUE_FULL);

    if(HIAI_OK != hiai_ret){
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] SendData batch %u failed! error code: %u", frameId, hiai_ret);
    }
    return hiai_ret;
}

/**
* @brief: run images on same side, all engine at same side
* @[return]: HIAI_StatusT
*/
HIAI_StatusT DataInputEngine::RunOnSameSide(){
    HIAI_StatusT ret = HIAI_OK;
    int totalCount = dataset_info_.size();
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DataInputEngine] run on %s, run %u images, batch size is %u", data_config_->target.c_str(), totalCount, data_config_->batchSize);
    for(int frameId = 0; frameId < totalCount; frameId++){
        //convert batch image infos to BatchImageParaWithScaleT
        std::shared_ptr<BatchImageParaWithScaleT> imageInfoBatch = std::make_shared<BatchImageParaWithScaleT>();
        if(imageInfoBatch == NULL){
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] make shared for BatchImageParaWithScaleT error!");
            return HIAI_ERROR;
        }
        NewImageParaT imgData;
        ret = makeImageInfo(&imgData, frameId);
        if(HIAI_OK != ret) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] Error: make image info frame id %u for batch id %u failed! Stop to send images!",frameId, frameId);
            return ret;
        }
        imageInfoBatch->v_img.push_back(imgData);
        imageInfoBatch->b_info.frame_ID.push_back(dataset_info_[frameId].id);
        //then send data
        ret = SendBatch(frameId, totalCount, imageInfoBatch);
        if(HIAI_OK != ret) {
            return ret;
        }
    }
    return  HIAI_OK;
}


/**
* @brief: Send Sentinel Image
*/
HIAI_StatusT DataInputEngine::SendSentinelImage()
{
    HIAI_StatusT ret = HIAI_OK;
    //all data send ok, then send a Sentinel info to other engine for end
    shared_ptr<BatchImageParaWithScaleT> image_handle = std::make_shared<BatchImageParaWithScaleT>();
    if(image_handle == NULL){
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DataInputEngine] make shared for BatchImageParaWithScaleT error!");
        return HIAI_ERROR;
    }
    ret = SendBatch(-1, 1, image_handle);
    return ret;
}

/**
* @ingroup hiaiengine
* @brief HIAI_DEFINE_PROCESS : Overloading Engine Process processing logic
* @[in]: Define an input port, an output port
*/
HIAI_IMPL_ENGINE_PROCESS("DataInputEngine", DataInputEngine, INPUT_SIZE)
{
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DataInputEngine] start process!");
    // gettimeofday(&start1, NULL);
    std::static_pointer_cast<string>(arg0);
    gettimeofday(&start1, NULL);
    HIAI_StatusT ret = HIAI_OK;

    ret = RunOnSameSide();

    //send sentinel image
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DataInputEngine] send sentinel image!");
    ret = SendSentinelImage();
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DataInputEngine] end process!");
    cout << "[DataInputEngine] end process!" << endl;
    return ret;
}
