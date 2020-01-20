#ifndef PostProcessEngine_H
#define PostProcessEngine_H

#include "hiaiengine/api.h"
#include "hiaiengine/ai_model_manager.h"
#include "hiaiengine/ai_types.h"
#include "hiaiengine/data_type.h"
#include "hiaiengine/engine.h"
#include "hiaiengine/multitype_queue.h"
#include "hiaiengine/data_type_reg.h"
#include "hiaiengine/ai_tensor.h"
#include "BatchImageParaWithScale.h"
#include <vector>
#include <utility>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include <hiaiengine/log.h>
#include <hiaiengine/ai_types.h>
#include "hiaiengine/ai_memory.h"

#define INPUT_SIZE 4
#define OUTPUT_SIZE 1
using hiai::Engine;
using namespace std;
using namespace cv;
using namespace hiai;

#define VDEC_MAX_CHANNEL 15; // 0~15 共16个通道

class ParamInfo
{
public:
	ParamInfo();
    int numBodyKeypoints;
    int numLimbs;
    std::vector<std::vector<int>> pairOrder;
    std::vector<std::vector<int>> limbConnectId;
    int minNumBodyPoints;
    float minHumanScore;
    float maxPafScore;
    int numIntermediatePoints;
    float minPortionGoodIntermediatePoints;
    float maxRatioToConnectKeypoints;
    float minIntermediaPointsScore;
    float minHeatmapValue;
};
inline ParamInfo::ParamInfo()
{
	numBodyKeypoints = 14;
    numLimbs = 13;
	int a[][2] = {{0, 1},  {1, 2},  {3, 4}, {4, 5}, {6, 7}, {7, 8}, {9, 10}, {10, 11}, {12, 13}, {13, 0}, {13, 3}, {13, 6}, {13, 9}};
    for(int i = 0; i < 13; i++){
        vector<int> vec;
        vec.push_back(a[i][0]);
        vec.push_back(a[i][1]);
        pairOrder.push_back(vec);
    }
	int b[][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}, {24, 25}};
    for(int i = 0; i < 13; i++){
        vector<int> vec;
        vec.push_back(b[i][0]);
        vec.push_back(b[i][1]);
        limbConnectId.push_back(vec);
    }
    minNumBodyPoints = 4;
    minHumanScore = 0.2;
    maxPafScore = 1.0;
    numIntermediatePoints = 10;
    minPortionGoodIntermediatePoints = 0.8;
    maxRatioToConnectKeypoints = 1.0;
    minIntermediaPointsScore = 0.0;
    minHeatmapValue = 0.04;
}


class JointInfo{
public:
	int row = 0;
	int col = 0;
	float score = 0.0;
	int count = 0;
	int type = 0;
};


class BestConnection{
public:
	int srcIndexInAllJoints = 0;
	int dstIndexInAllJoints = 0;
	float score = 0.0;
	int srcIndexInAMap = 0;
	int dstIndexInAMap = 0;
};

class PostProcessEngine : public Engine {
public:
    ~PostProcessEngine();
    PostProcessEngine() :input_que_(INPUT_SIZE) {}
    HIAI_StatusT Init(const AIConfig& config, const std::vector<AIModelDescription>& model_desc);
    // OutputT PostProcess(shared_ptr<EngineTransT> tran_data0);

    vector<vector<JointInfo>> FindAllJoints(ParamInfo param, const int factor, bool boolRefineCenter, OutputT out);

    vector<JointInfo> AddType(vector<vector<JointInfo>> JointsInfoInAllMaps);

    vector<int> Linspace(int srcPoint, int dstPoint, int amount);

    float GetValue(int row, int col, int channel, vector<OutputT> pafmapOut);

    vector<vector<BestConnection>> FindConnectedJoints(vector<vector<JointInfo>> JointsInfoInAllMaps, ParamInfo param, vector<OutputT> pafmapOut);

    vector<vector<float>> GroupLimbsOfSamePerson(ParamInfo param, vector<vector<BestConnection>> ConnectedLimbsInfo, vector<JointInfo> alljointsInfo);

    vector<float> genHumansFromJointsInfo(ParamInfo param, vector<JointInfo> alljointsInfo, vector<vector<float>> allPersonsInfo);
    HIAI_DEFINE_PROCESS(INPUT_SIZE, OUTPUT_SIZE)

private:
    MultiTypeQueue input_que_;
};

#endif
