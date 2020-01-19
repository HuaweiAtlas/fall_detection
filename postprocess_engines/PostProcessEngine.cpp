#include "PostProcessEngine.h"
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
#include <fstream>
#include <stdlib.h>
#include <sys/stat.h>
#include <sstream>
#include <fcntl.h>

#define PADSIZE 2
#define IMAGEW 384
#define IMAGEH 288
#define WSIZE 96
#define HSIZE 72
#define PERSON_INFO_SIZE 20
#define PERSON_SCORE_INDEX 18
#define PERSON_JOINTS_COUNT_INDEX 19
#define INTERMEDIA_POINT_AMOUNT 10
#define HEATMAP_SIZE 14
#define PAFMAP_SIZE 26

uint64_t PostProcessEngine_count = 0;

HIAI_StatusT PostProcessEngine::Init(const AIConfig& config, const std::vector<AIModelDescription>& model_desc)
{
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "robot_dog: PostProcessEngine init");
    return HIAI_OK;
}

PostProcessEngine::~PostProcessEngine()
{
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "robot_dog: PostProcessEngine exit !");
}

/**
* @brief: get all joints in all heatmaps
* @in: param: include parameters, numBodyKeypoints:14,minHeatmapValue:0.075
       factor: resize factor, because original image size is 368*368, heatmap size is 92*92,the factor is 4
       boolRefineCenter: whether refine the position of max value, if boolRefineCenter is true, calculate the offsets and refine the position
            of max value after resize. 
       out: heatmaps pointer.we can get heatmaps from momery according to this pointer.
* @return: jointsPerType: [14,x],14 represent amount of heatmaps, x represent amount of peaks in each heatmap, each peak represent a 
            object(JointInfo):row,col,score,count,type.
*/
vector<vector<JointInfo>> PostProcessEngine::FindAllJoints(ParamInfo param, const int factor, bool boolRefineCenter, OutputT out){
    vector<vector<JointInfo>> jointsPerType;
    int cntJoints = 0;
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3)); // getstructuringElement is a opencv function to define kernel type for dilate.
    // ========================get all jointsInfo in each heatmap,the amount of heatmaps is equal to numBodyKeypoints===================
    for(int i = 0; i < param.numBodyKeypoints; ++i){
        // =====step1:get correspending data for each heatmap from memory and reshape into 2 dimension mat,then calculate max value=============
        vector<float> heatmap((float*)(out.data.get()) + i*HSIZE*WSIZE, (float*)(out.data.get()) + (i + 1)*HSIZE*WSIZE);
        Mat heatMat = Mat(heatmap).reshape(0, HSIZE);
        Mat maxMat;
        dilate(heatMat, maxMat, kernel); // dilate is a opencv function to calculate max value in a relative field and replace the original value
        // =====step2:get the position of max value in heatmap,then push (x,y) into a vector=======================
        vector<vector<int>> peaksOneHeatmap;
        vector<int> coords(2);
        for(int i = 0; i < HSIZE; i++){
            for(int j = 0; j < WSIZE; j++){
                // compare the original mat and max mat,if the value is equal and value is beyond minvalue,
                // then push the coordialtes in a vector
                if(heatMat.at<float>(i, j) == maxMat.at<float>(i, j) && heatMat.at<float>(i, j) > param.minHeatmapValue){
                    coords[0] = i;
                    coords[1] = j;
                    peaksOneHeatmap.push_back(coords);
                }
            }
        }
        // =====================for each max value(peak),store it's information in a vector============================================
        int jointsAmount = peaksOneHeatmap.size();
        // cout << "jointsAmount:" << jointsAmount <<endl;
        vector<JointInfo> jointsInfoInAMap(jointsAmount); // initialize peak*5 vector to store information of max value
        for(int i = 0; i < jointsAmount; i++){
            vector<int> peak = peaksOneHeatmap[i];
            vector<float> offSets(2);
            float peakScore;
            if(boolRefineCenter == true){
                // =====step3:get a relative filed around max value into a mat and resize mat with factor==========
                // calculate patch size, x represent rows, y represent cols;
                int xmin = max(0, peak[0] - PADSIZE);
                int ymin = max(0, peak[1] - PADSIZE);
                int xmax = min(HSIZE - 1, peak[0] + PADSIZE);
                int ymax = min(WSIZE - 1, peak[1] + PADSIZE);
                int rowSize = xmax + 1 - xmin; 
                int colSize = ymax + 1 - ymin; 
                // get patchMat from original heatMat
                Mat patchMat(rowSize, colSize, CV_32FC1);
                for(int row = xmin; row < xmax + 1; ++row)
                    for(int col = ymin; col < ymax + 1; ++col){
                        patchMat.at<float>(row-xmin, col-ymin) = heatMat.at<float>(row, col);
                    }
                // resize the patchMat with factor
                Mat resizePatchMat;
                resize(patchMat, resizePatchMat, Size(round(factor*colSize), round(factor*rowSize)), 0, 0, INTER_CUBIC);
                // =====step4:calcaulte the max point offsets after resize===============================
                // get original max point position after resize 
                float maxLocAfterResizeX = (peak[0] - xmin + 0.5)*factor - 0.5; 
                float maxLocAfterResizeY = (peak[1] - ymin + 0.5)*factor - 0.5; 
                // minMaxLoc is a opencv function to calculate max and min value and it's position in a mat 
                double minValue = 0;
                double maxValue = 0;
                Point maxLocation, minLocation;
                minMaxLoc(resizePatchMat, &minValue, &maxValue, &minLocation, &maxLocation); 
                // calculate offsets and store the max value in resizedPatchMat as peakscore
                offSets[0] = maxLocation.x - maxLocAfterResizeX;
                offSets[1] = maxLocation.y - maxLocAfterResizeY; 
                peakScore = resizePatchMat.at<float>(maxLocation.y, maxLocation.x);
            }
            else{
                peakScore = heatMat.at<float>(peak[0], peak[1]);
            }
            // =====step5:refine the position of max value and store information in a struct===============
            JointInfo jointInfo;
            jointInfo.row = floor((peak[0] + 0.5)*factor - 0.5 + offSets[0]);
            jointInfo.col = floor((peak[1] + 0.5)*factor - 0.5 + offSets[1]);
            jointInfo.score = peakScore;
            jointInfo.count = cntJoints;
            jointInfo.type = 0;
            jointsInfoInAMap[i] = jointInfo;
            cntJoints += 1;
        }
        // store a heatmap's jointsInfo
        jointsPerType.push_back(jointsInfoInAMap);
    }
    return jointsPerType;
}


/**
* @brief: add joint type in each joint
* @in: jointsPerType: [14,x],14 represent amount of heatmaps, x represent amount of peaks in each heatmap, each peak represent a 
            object(JointInfo):row,col,score,count,type
* @return: allJointsInfo: .size = x,x represent amount of joints in all heatmaps, each element is a object(JointInfo)
*/
vector<JointInfo> PostProcessEngine::AddType(vector<vector<JointInfo>> jointsPerType){
    vector<JointInfo> allJointsInfo;
    // ==============for each heatmap====================
    for(int i = 0; i < int(jointsPerType.size()); i++){
        int jointsAmount = jointsPerType[i].size();
        // if there is no joints in heatmap,continue
        if(jointsAmount == 0){
            continue; 
        }
        // ==============for each joint================
        for(int j = 0; j < jointsAmount; ++j){
            JointInfo jointInfo = jointsPerType[i][j];
            jointInfo.type = i;
            allJointsInfo.push_back(jointInfo);
            // cout << "joints:" << jointInfo.row << " " <<jointInfo.col<< " " << jointInfo.score <<" "<< jointInfo.count << " "<< jointInfo.type << endl;
        }
    }
    return allJointsInfo;
}

/**
* @brief: get ten number between a and b
* @in: a: start number
        b: end number
        c: amount of number
* @return: vec
*/
vector<int> PostProcessEngine::Linspace(int srcPoint, int dstPoint, int amount){
    vector<int>vec(10);
    float delta = (dstPoint - srcPoint) / (amount - 1);
    for(int i = 0; i < amount; i++){
        vec[i] = round(srcPoint + i*delta);
    }
    return vec;
}

/**
* @brief: get Index in memory
* @in: 
* @return: int
*/
float PostProcessEngine::GetValue(int row, int col, int channel, vector<OutputT> pafmapOut){
    int i = channel/7;
    int index;
    OutputT out = pafmapOut[i];
    if(i == 3){
        index = row*5*384 + col*5 + channel%7;
    }else{
        index = row*7*384 + col*7 + channel%7;
    }
    float value = *((float*)(out.data.get()) + index); 
    return value;
}

/**
* @brief: according to limbtype and pairorder, get src heatmap and dst heatmap. For each joint in src heatmap and each joint in dst heatmap,
          calculate three criterion,store the most matched pair joints
* @in: jointsPerType:[14,x],14 represent amount of heatmaps, x represent amount of peaks in each heatmap, each peak represent a 
            object(JointInfo):row,col,score,count,type.
       param: include parameters,
       out:pafmap pointer,we can get pafmap from memory according to this pointer
* @return: connectedLimbsPerType: [13,x],13 represent amount of limbtype,x represent limb amount in each type,each element is a object[BestConnection]
             srcIndexInAllJoints,dstIndexInAllJoints,score,srcIndexInAMap,dstsIndexInAMap
*/
vector<vector<BestConnection>> PostProcessEngine::FindConnectedJoints(
                                        vector<vector<JointInfo>>jointsPerType, 
                                        ParamInfo param, 
                                        vector<OutputT> pafmapOut){
    double distanceThreshW = IMAGEW * param.maxRatioToConnectKeypoints;
    double distanceThreshH = IMAGEH * param.maxRatioToConnectKeypoints; //368*0.5=184
    int minPointNumber = param.minPortionGoodIntermediatePoints*param.numIntermediatePoints;

    vector<vector<BestConnection>> connectedLimbsPerType;
    // =========================for each libmtype============================================
    for(int limbType = 0; limbType < param.numLimbs; limbType++){
        // =====step1:get joints information in src heatmap and dst heatmap according to limbtype and pairorder==========
        int srcJointsType = param.pairOrder[limbType][0];
        int dstJointsType = param.pairOrder[limbType][1];
        vector<JointInfo> srcJointsInfo = jointsPerType[srcJointsType];
        vector<JointInfo> dstJointsInfo = jointsPerType[dstJointsType];
        int srcJointsAmount = srcJointsInfo.size();
        int dstJointsAmount = dstJointsInfo.size();
        // if there is no joint in src heatmap or dst heatmap
        if(srcJointsAmount == 0 || dstJointsAmount == 0){
            vector<BestConnection> connections;
            connectedLimbsPerType.push_back(connections);
        }
        // if there are joints in src heatmap and dst heatmap, and maybe more than 1 connections
        else{
            vector<BestConnection> connections;
            // =====step2:get srcIndex and dstIndex in pafmap according to limbtype and limbConnectId======
            int srcPafmapIndex = param.limbConnectId[limbType][0]; 
            int dstPafmapIndex = param.limbConnectId[limbType][1]; 
            
            // =========================for each joint in src heatmap==================
            for(int srcIndex = 0; srcIndex < srcJointsAmount; srcIndex++){
                float bestScore = 0.0;
                BestConnection bestConnection;
                int srcJointRow = srcJointsInfo[srcIndex].row;
                int srcJointCol = srcJointsInfo[srcIndex].col;
                // =======================for each joint in dst heatmap=====================
                for(int dstIndex = 0; dstIndex < dstJointsAmount; dstIndex++){
                    int dstJointRow = dstJointsInfo[dstIndex].row;
                    int dstJointCol = dstJointsInfo[dstIndex].col;

                    // ======step3:calculate unit vector between srcJoint and dstJoint==========
                    int distanceRow = dstJointRow - srcJointRow;
                    int distanceCol = dstJointCol - srcJointCol;
                    // if distance > Thresh,continue to next joint
                    if( distanceRow > distanceThreshH || distanceCol > distanceThreshW){
                        continue;
                    }
                    float limbDistance = sqrt(distanceRow*distanceRow + distanceCol*distanceCol) + 1e-8; 
                    float unitRow = distanceRow / limbDistance;
                    float unitCol = distanceCol / limbDistance;

                    // =====step4:get 10 intermediate points between srcJoint and dstJoint==============
                    vector<int> rowIntermdiaPointVec = Linspace(srcJointRow, dstJointRow, INTERMEDIA_POINT_AMOUNT);
                    vector<int> colIntermdiaPointVec = Linspace(srcJointCol, dstJointCol, INTERMEDIA_POINT_AMOUNT);

                    vector<float> intermediaPointScoreVec;
                    float totalScore = 0.0;
                    int overCount = 0;
                    // ======step5:for each intermediate point between srcjoint and dst dstjoint,get 10 value in memory
                    // according to 10 intermediate points,then multiply it with unit vector to get 10 score=========
                    for(int i = 0; i < INTERMEDIA_POINT_AMOUNT; i++){
                        // float rowValue = pafmaps[srcPafmapIndex].at<float>(rowIntermdiaPointVec[i], colIntermdiaPointVec[i]);
                        // float colValue = pafmaps[dstPafmapIndex].at<float>(rowIntermdiaPointVec[i], colIntermdiaPointVec[i]);
                        float rowValue = GetValue(rowIntermdiaPointVec[i], 
                                                colIntermdiaPointVec[i], 
                                                srcPafmapIndex,
                                                pafmapOut);
                        float colValue = GetValue(rowIntermdiaPointVec[i], 
                                                colIntermdiaPointVec[i],
                                                dstPafmapIndex,
                                                pafmapOut);
                        // float rowValue = *((float*)(input_data.get()) + srcIndexInMemory);
                        // float colValue = *((float*)(input_data.get()) + dstIndexInMemory);
                        // float rowValue = *((float*)(input_data.get()) + srcIndexInMemory);
                        // float colValue = *((float*)(input_data.get()) + dstIndexInMemory);
                        float intermediaPointScore = rowValue*unitCol + colValue*unitRow;
                        totalScore += intermediaPointScore;
                        if (intermediaPointScore > param.minIntermediaPointsScore){
                            overCount += 1;
                        }
                    }
                    // =====step6:calculate three standard,if three standard all are true,it means that srcJoint and dstJoint is connected.
                    // but we only store the most matched dstJoint==========================
                    bool criterion1 = (overCount > minPointNumber) ? true : false;
                    float scorePenalizingDistance = totalScore/INTERMEDIA_POINT_AMOUNT + min(distanceThreshW / limbDistance - 1, 0.0);
                    bool criterion2 = (scorePenalizingDistance > 0) ? true : false;
                    bool criterion3 = (scorePenalizingDistance > bestScore) ? true : false;
                    if(criterion1 == true && criterion2 == true && criterion3 == true){
                        bestScore = scorePenalizingDistance; 
                        bestConnection.srcIndexInAllJoints = srcJointsInfo[srcIndex].count;
                        bestConnection.dstIndexInAllJoints = dstJointsInfo[dstIndex].count;
                        bestConnection.score = scorePenalizingDistance;
                        bestConnection.srcIndexInAMap = srcIndex;
                        bestConnection.dstIndexInAMap = dstIndex;
                        if(bestScore > param.maxPafScore){
                            break;
                        }
                    }
                }
                // =====step7:store the most matched srcJoint and dstJoint=================
                if(bestConnection.score - 0.0 > (10^(-6))){
                    // cout << bestConnection.srcIndexInAllJoints << " "<< bestConnection.dstIndexInAllJoints<< " " << bestConnection.score<< " "
                    // << bestConnection.srcIndexInAMap << " "<< bestConnection.dstIndexInAMap << endl;
                    connections.push_back(bestConnection);
                }
            }
            // for this limbtype,we have calculate all srcJoints and dstJoints and store the connections,then go to next limbtype
            connectedLimbsPerType.push_back(connections);
        }
    }
    return connectedLimbsPerType;
}



/**
* @brief: we group limb information of same person to get a person's joint information
* @in: param: include parameters,
       connectedLimbsPerType:[13,x],13 represent amount of limbtype,x represent limb amount in each type,each element is a object[BestConnection]
                srcIndexInAllJoints,dstIndexInAllJoints,score,srcIndexInAMap,dstsIndexInAMap
       jointsInfo: allJointsInfo: .size = x,x represent amount of joints in all heatmaps, each element is a object(JointInfo)
* @return: allPersonsInfo : [x,20],x represent the amount of person,20 represent [indexInAllJoints]*14 +[-1]*4 + score + JointCount
                in the paper,there are 18 joints,but we only have 14 joints,so we have -1*4 here.
*/
vector<vector<float>> PostProcessEngine::GroupLimbsOfSamePerson(ParamInfo param, vector<vector<BestConnection>> connectedLimbsPerType, \
                            vector<JointInfo> allJointsInfo){
    vector<vector<float>> allPersonsInfo;
    // =====================for each limb type===================
    for(int limbType = 0; limbType < param.numLimbs; limbType++){
        int srcJointsType = param.pairOrder[limbType][0];  
        int dstJointsType = param.pairOrder[limbType][1];
        // =====step1:get limbsInfo in per limbtype===================
        vector<BestConnection> limbsInfo = connectedLimbsPerType[limbType]; 
        
        int limbsAmount = limbsInfo.size();
        if(limbsAmount == 0){
            continue;
        }
        // =====================for each limb in a limb type===================
        for(int i = 0; i < limbsAmount; i++){
            BestConnection limbInfo = limbsInfo[i]; 
            vector<int> assocPersonIndex;
            // =====step2:compare the new limb with detected person,if they have connected joint,store the index for further detect.
            if(allPersonsInfo.size() != 0){
                for(int personIndex = 0; personIndex < int(allPersonsInfo.size()); personIndex++){
                    vector<float> onePersonInfo = allPersonsInfo[personIndex]; 
                    if(onePersonInfo[srcJointsType] == limbInfo.srcIndexInAllJoints || \
                       onePersonInfo[dstJointsType] == limbInfo.dstIndexInAllJoints){
                        assocPersonIndex.push_back(personIndex);
                    }
                }
            }
            // =====step3:according to the connected size,we use different method to deal with the new limb=============
            int personAmount = assocPersonIndex.size();
            // if the new limb is connected with one detected person,merge it with that person
            if(personAmount == 1){
                // when we get here,it means that either the new limb's src jointType is same as this person's src joint type or 
                // limb's dst type is same as person's dst joint type,so if limb's dst jointType is not same as person's
                // dst jointType,we merge it with person.
                if(allPersonsInfo[assocPersonIndex[0]][dstJointsType] != limbInfo.dstIndexInAllJoints){
                    allPersonsInfo[assocPersonIndex[0]][dstJointsType] = limbInfo.dstIndexInAllJoints; 
                    allPersonsInfo[assocPersonIndex[0]][PERSON_SCORE_INDEX] += (allJointsInfo[limbInfo.dstIndexInAllJoints].score + \
																			limbInfo.score);
                    allPersonsInfo[assocPersonIndex[0]][PERSON_JOINTS_COUNT_INDEX] += 1.0;
                }
            }
            // if the new limb is connected with two detected person,personAmount ==2
            else if(personAmount == 2){
                vector<float> membership;
                int firstPersonIndex = assocPersonIndex[0];
                int secondPersonIndex = assocPersonIndex[1];
                // compare the every joint element,we use PERSON_SCORE_INDEX because PERSON_SCORE_INDEX = PERSON_INFO_SIZE -2.It's equal to joint amount.
                // If both person have same joint ,store a ture for further judgement
                for(int i = 0; i < PERSON_SCORE_INDEX; i++){
                    if(allPersonsInfo[firstPersonIndex][i] >= 0 && allPersonsInfo[secondPersonIndex][i] >= 0){
                        membership.push_back(true);
                    }
                }
                if(membership.size() == 0){
                    // if both person have no same joints connected,merge them into a single person
                    for(int i = 0; i < PERSON_SCORE_INDEX; i++){
                        // because we initial the vector as -1,so we add 1 here to ensure right jointsIndex.
                        allPersonsInfo[firstPersonIndex][i] += (allPersonsInfo[secondPersonIndex][i] + 1);
                    }
                    allPersonsInfo[firstPersonIndex][PERSON_SCORE_INDEX] += (allPersonsInfo[secondPersonIndex][PERSON_SCORE_INDEX] + limbInfo.score);
                    allPersonsInfo[firstPersonIndex][PERSON_JOINTS_COUNT_INDEX] += allPersonsInfo[secondPersonIndex][PERSON_JOINTS_COUNT_INDEX];
                    // we have merged the second person's information into first person,so we erase it in allpersoninfo.
                    allPersonsInfo.erase(allPersonsInfo.begin() + secondPersonIndex);
                }
                else{
                    // if both person have same joints connected,same case as personAmount == 1
                    allPersonsInfo[firstPersonIndex][dstJointsType] = limbInfo.dstIndexInAllJoints;
                    allPersonsInfo[firstPersonIndex][PERSON_SCORE_INDEX] += (allJointsInfo[limbInfo.dstIndexInAllJoints].score + \
                                                                     limbInfo.score);
                    allPersonsInfo[firstPersonIndex][PERSON_JOINTS_COUNT_INDEX] += 1.0;
                }
            }
            // if the new limb is not connected with any detected person, create a new person information.
            else {
                vector<float> person(PERSON_INFO_SIZE, -1);
                person[srcJointsType] = limbInfo.srcIndexInAllJoints; 
                person[dstJointsType] = limbInfo.dstIndexInAllJoints;
                person[PERSON_SCORE_INDEX] = allJointsInfo[limbInfo.srcIndexInAllJoints].score + \
                                             allJointsInfo[limbInfo.dstIndexInAllJoints].score + \
                                             limbInfo.score; 
                person[PERSON_JOINTS_COUNT_INDEX] = 2.0; 
                allPersonsInfo.push_back(person); 
            }
        }
    }
    // =====step4:we use some condition to earse some person's information========================
    // HIAI_ENGINE_LOG(HIAI_IDE_ERROR,"allPersonsInfo.size: %d",allPersonsInfo.size());
    vector<int>delete_id;
    for(int personIndex = 0; personIndex < int(allPersonsInfo.size()); personIndex++){
        vector<float> onePersonInfo = allPersonsInfo[personIndex];
        if((int(onePersonInfo[PERSON_JOINTS_COUNT_INDEX]) < param.minNumBodyPoints) ||
            (onePersonInfo[PERSON_SCORE_INDEX] / onePersonInfo[PERSON_JOINTS_COUNT_INDEX] < param.minHumanScore)){
                delete_id.insert(delete_id.begin(),personIndex);
        }
    }
    for(int i = 0; i < int(delete_id.size()); i ++){
        int id = delete_id[i];
        allPersonsInfo.erase(allPersonsInfo.begin() + id);
    }
    // HIAI_ENGINE_LOG(HIAI_IDE_ERROR,"allPersonsInfo.size: %d",allPersonsInfo.size());
    return allPersonsInfo;
}


/**
* @brief: generate human from person's jointIndex and jointsInfo. 
* @in:allJointsInfo: .size = x,x represent amount of joints in all heatmaps, each element is a object(JointInfo)
* @return: human joint info
*/
vector<float> PostProcessEngine::genHumansFromJointsInfo(ParamInfo param, vector<JointInfo> allJointsInfo, \
                                                      vector<vector<float>> allPersonsInfo){
    vector<float> human;
    // =================for each person====================
    for(int personIndex = 0; personIndex < int(allPersonsInfo.size()); personIndex++){
        vector<float> onePersonInfo = allPersonsInfo[personIndex];
        for(int jointType = 0; jointType < HEATMAP_SIZE; jointType++){
            if(int(onePersonInfo[jointType]) == -1){
                human.push_back(0.0);
                human.push_back(0.0);
                human.push_back(0.0);
                continue;
            }
            JointInfo jointInfo = allJointsInfo[onePersonInfo[jointType]];
            human.push_back(jointInfo.row);
            human.push_back(jointInfo.col);
            human.push_back(jointInfo.score);
        }
    }
    return human;
}


HIAI_IMPL_ENGINE_PROCESS("PostProcessEngine", PostProcessEngine, INPUT_SIZE)
{
    HIAI_StatusT ret = HIAI_OK;

    // struct timeval start1;
    // struct timeval stop1;
    // memset(&start1, 0, sizeof(start1));
    // memset(&stop1, 0, sizeof(stop1));
    // gettimeofday(&start1, NULL);

    // 从默认参数中获取输入engine的数据，转换成定义的格式。
    std::shared_ptr<EngineTransT> tran_data0 = std::static_pointer_cast<EngineTransT>(arg0);
    std::shared_ptr<EngineTransT> tran_data1 = std::static_pointer_cast<EngineTransT>(arg1);
    std::shared_ptr<EngineTransT> tran_data2 = std::static_pointer_cast<EngineTransT>(arg2);
    std::shared_ptr<EngineTransT> tran_data3 = std::static_pointer_cast<EngineTransT>(arg3);
    input_que_.PushData(0, arg0);
    input_que_.PushData(1, arg1);
    input_que_.PushData(2, arg2);
    input_que_.PushData(3, arg3);
    if (!input_que_.PopAllData(tran_data0, tran_data1, tran_data2, tran_data3))
    {
        HIAI_ENGINE_LOG("fail to get all  message");
        return HIAI_ERROR;
    }
    cout << "postprocess:" << tran_data0->msg << " "<< tran_data1->msg << " "<< tran_data2->msg << " "<< tran_data3->msg <<endl;
    BatchImageParaWithScaleT image_handle = {tran_data0->b_info, tran_data0->v_img};
    if (isSentinelImage(std::make_shared<BatchImageParaWithScaleT>(image_handle)))
    {   
        HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[PostProcessEngine]sentinel Image, process over.");
        HIAI_StatusT hiaiRet = HIAI_OK;
        do{
            hiaiRet = SendData(0, "EngineTransT", std::static_pointer_cast<void>(tran_data0));
            if (HIAI_OK != hiaiRet) {
                if (HIAI_ENGINE_NULL_POINTER == hiaiRet || HIAI_HDC_SEND_MSG_ERROR == hiaiRet || HIAI_HDC_SEND_ERROR == hiaiRet
                    || HIAI_GRAPH_SRC_PORT_NOT_EXIST == hiaiRet || HIAI_GRAPH_ENGINE_NOT_EXIST == hiaiRet) {
                    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[PostProcessEngine] SendData error[%d], break.", hiaiRet);
                    break;
                }
                HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[PostProcessEngine] SendData return value[%d] not OK, sleep 200ms", hiaiRet);
                usleep(SEND_DATA_INTERVAL_MS);
            }
        } while (HIAI_OK != hiaiRet);
        return hiaiRet;
    }
    // cout << "postprocess0:" << tran_data0->msg << endl;
    // cout << "postprocess1:" << tran_data1->msg << endl;
    // cout << "postprocess2:" << tran_data2->msg << endl;
    // cout << "postprocess3:" << tran_data3->msg << endl;
    tran_data0->msg = "postprocessed results";
 
    //Step 1: find all joints in the image (organized by joint type: [0]=nose, [1]=neck...)
    int scale = IMAGEW/WSIZE;
    ParamInfo param;
    vector<vector<JointInfo>> JointsInfoAllMap = FindAllJoints(param, scale, true, tran_data0->output_data_vec[0]); 

    //Step 2: add type information in each jointInfo
    vector<JointInfo> allJointsInfo = AddType(JointsInfoAllMap);
    
    OutputT out0 = tran_data0->output_data_vec[1];
    OutputT out1 = tran_data1->output_data_vec[1];
    OutputT out2 = tran_data2->output_data_vec[1];
    OutputT out3 = tran_data3->output_data_vec[1];
    vector<OutputT> pafmapOut;
    pafmapOut.push_back(out0);
    pafmapOut.push_back(out1);
    pafmapOut.push_back(out2);
    pafmapOut.push_back(out3);
    // int input_size = out0.size + out1.size + out2.size + out3.size;
    // shared_ptr<u_int8_t> input_data(new u_int8_t[input_size]);
    // memcpy_s(input_data.get(), out0.size, out0.data.get(), out0.size);
    // memcpy_s(input_data.get() + out0.size, out1.size, out1.data.get(), out1.size);
    // memcpy_s(input_data.get() + out0.size + out1.size, out2.size, out2.data.get(), out2.size);
    // memcpy_s(input_data.get() + out0.size + out1.size + out2.size, out3.size, out3.data.get(), out3.size);

    // cout << *((float*)(input_data.get() + 5*7*384 + 3*7 + 4)) << endl;
    // cout << *((float*)(input_data.get() + 5*7*384 + 3*7 + 4 + 384*288*7)) << endl;
    // cout << *((float*)(input_data.get() + 5*7*384 + 3*7 + 4 + 384*288*14)) << endl;
    // cout << *((float*)(input_data.get() + 5*5*384 + 3*7 + 4 + 384*288*21)) << endl;
    // OutputT out = tran_data0->output_data_vec[1];
    //Step 3: find which joints go together to form limbs (which wrists go with which elbows)
    vector<vector<BestConnection>> connectedLimbsPerType = FindConnectedJoints(JointsInfoAllMap, param, pafmapOut);

    // //Step 4: associate limbs that belong to the same person
    vector<vector<float>> allPersonsInfo = GroupLimbsOfSamePerson(param, connectedLimbsPerType, allJointsInfo);
    // //Step 5 Generate human from joint
    vector<float> humans = genHumansFromJointsInfo(param, allJointsInfo, allPersonsInfo);

    // OutputT out1;
    int buffer_size = sizeof(float)*humans.size();
    shared_ptr<u_int8_t> human_data(new u_int8_t[buffer_size]);
    memcpy_s(human_data.get(), buffer_size, humans.data(), buffer_size);
    OutputT output_data;
    output_data.size = buffer_size;
    output_data.data = human_data;
    output_data.name = "persons";
    tran_data0->output_data_vec[0] = output_data;
    tran_data0->output_data_vec.pop_back();

    ret = SendData(0, "EngineTransT", std::static_pointer_cast<void>(tran_data0));

    // gettimeofday(&stop1, NULL);
    // double time_used1 = (stop1.tv_sec - start1.tv_sec) *1000+(stop1.tv_usec - start1.tv_usec) / 1000.0;
    // cout << "end,postprocess time use is:" << time_used1 << endl;

    if (ret != HIAI_OK)
    {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "robot_dog: PostProcessEngine SendData to device fail! ret = %d", ret);
        return ret;
    }
    PostProcessEngine_count++;
    // cout << "[PostProcessEngine] end process!" << endl;
    return HIAI_OK;
}
