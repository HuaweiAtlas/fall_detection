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

#include "Postprocess.h"

#include <cstdint>
#include <memory>
#include <sstream>

#include "hiaiengine/log.h"
#include "ascenddk/ascend_ezdvpp/dvpp_process.h"
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


HIAI_StatusT Postprocess::Init(
    const hiai::AIConfig& config,
    const vector<hiai::AIModelDescription>& model_desc) {
  HIAI_ENGINE_LOG("End initialize!");
  return HIAI_OK;
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
vector<vector<JointInfo>> Postprocess::FindAllJoints(ParamInfo param, const int factor, bool boolRefineCenter, OutputT out){
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
vector<JointInfo> Postprocess::AddType(vector<vector<JointInfo>> jointsPerType){
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
vector<int> Postprocess::Linspace(int srcPoint, int dstPoint, int amount){
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
float Postprocess::GetValue(int row, int col, int channel, vector<OutputT> pafmapOut){
    int i = channel/7;
    int index;
    OutputT out = pafmapOut[0];
    // if(i == 3){
    //     index = row*5*384 + col*5 + channel%7;
    // }else{
    //     index = row*7*384 + col*7 + channel%7;
    // }
    index = row*26*384 + col*26 + channel;
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
vector<vector<BestConnection>> Postprocess::FindConnectedJoints(
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
vector<vector<float>> Postprocess::GroupLimbsOfSamePerson(ParamInfo param, vector<vector<BestConnection>> connectedLimbsPerType, \
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
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR,"allPersonsInfo.size: %d",allPersonsInfo.size());
    return allPersonsInfo;
}


/**
* @brief: generate human from person's jointIndex and jointsInfo. 
* @in:allJointsInfo: .size = x,x represent amount of joints in all heatmaps, each element is a object(JointInfo)
* @return: human joint info
*/
vector<float> Postprocess::genHumansFromJointsInfo(ParamInfo param, vector<JointInfo> allJointsInfo, \
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

void Postprocess::SendResult(
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

HIAI_StatusT Postprocess::Recognition(
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
//   std::shared_ptr<FaceRecognitionInfo> image_handle = std::static_pointer_cast<FaceRecognitionInfo>(arg0);
//   std::shared_ptr<FaceRecognitionInfo> tran_data1 = std::static_pointer_cast<FaceRecognitionInfo>(arg1);
//   std::shared_ptr<FaceRecognitionInfo> tran_data2 = std::static_pointer_cast<FaceRecognitionInfo>(arg2);
//   std::shared_ptr<FaceRecognitionInfo> tran_data3 = std::static_pointer_cast<FaceRecognitionInfo>(arg3);
//   input_que_.PushData(0, arg0);
//   input_que_.PushData(1, arg1);
//   input_que_.PushData(2, arg2);
//   input_que_.PushData(3, arg3);
//   if (!input_que_.PopAllData(image_handle, tran_data1, tran_data2, tran_data3))
//   {
//     HIAI_ENGINE_LOG("fail to get all  message");
//     return HIAI_ERROR;
//   }

//   image_handle->msg = "postprocessed results";
 
  //Step 1: find all joints in the image (organized by joint type: [0]=nose, [1]=neck...)
  int scale = IMAGEW/WSIZE;
  ParamInfo param;
  if (image_handle->frame.image_source == 1) {
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "postprocess: heatmap size is %d", image_handle->output_data_vec[0].size);
  }
  vector<vector<JointInfo>> JointsInfoAllMap = FindAllJoints(param, scale, true, image_handle->output_data_vec[0]); 

  //Step 2: add type information in each jointInfo
  vector<JointInfo> allJointsInfo = AddType(JointsInfoAllMap);
    
  OutputT out0 = image_handle->output_data_vec[1];
//   OutputT out1 = tran_data1->output_data_vec[1];
//   OutputT out2 = tran_data2->output_data_vec[1];
//   OutputT out3 = tran_data3->output_data_vec[1];
  vector<OutputT> pafmapOut;
  pafmapOut.push_back(out0);
//   pafmapOut.push_back(out1);
//   pafmapOut.push_back(out2);
//   pafmapOut.push_back(out3);
  //Step 3: find which joints go together to form limbs (which wrists go with which elbows)
  vector<vector<BestConnection>> connectedLimbsPerType = FindConnectedJoints(JointsInfoAllMap, param, pafmapOut);

  // //Step 4: associate limbs that belong to the same person
  vector<vector<float>> allPersonsInfo = GroupLimbsOfSamePerson(param, connectedLimbsPerType, allJointsInfo);
  // //Step 5 Generate human from joint
  vector<float> humans = genHumansFromJointsInfo(param, allJointsInfo, allPersonsInfo);
//   HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "humans size is %d", humans.size());
  // OutputT out1;
  HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "postprocess finish");
  int buffer_size = sizeof(float)*humans.size();
  shared_ptr<u_int8_t> human_data(new u_int8_t[buffer_size]);
  memcpy_s(human_data.get(), buffer_size, humans.data(), buffer_size);
  OutputT output_data;
  output_data.size = buffer_size;
  output_data.data = human_data;
  output_data.name = "persons";
  image_handle->output_data_vec[0] = output_data;
//   image_handle->output_data_vec.pop_back();
  // send result
  SendResult(image_handle);
  HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Send finish");
  return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("Postprocess", Postprocess, INPUT_SIZE) {
  HIAI_StatusT ret = HIAI_OK;

  // deal arg0 (engine only have one input)
  if (arg0 != nullptr) {
    HIAI_ENGINE_LOG("begin to deal face_recognition!");
    shared_ptr<FaceRecognitionInfo> image_handle = static_pointer_cast<
        FaceRecognitionInfo>(arg0);

    // deal data from camera
    if (image_handle->frame.image_source == 0) {
      HIAI_ENGINE_LOG("post process dealing data from camera.");
      SendResult(image_handle);
      return HIAI_OK;
    }

    // deal data from register
    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "post process dealing data from register.");
    ret = Recognition(image_handle);
  }
//   HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "postprocess start ");
//   ret = SendData(0, "FaceRecognitionInfo", std::static_pointer_cast<void>(image_handle));
  return ret;
}
