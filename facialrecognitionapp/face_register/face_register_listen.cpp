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

#include <memory>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <regex>
#include "hiaiengine/log.h"
#include "hiaiengine/data_type_reg.h"
#include "face_register_listen.h"
#include "presenter_channels.h"


using hiai::Engine;
using namespace hiai;
using namespace std;
using namespace ascend::presenter;
using namespace ascend::presenter::facial_recognition;

int SendDataByGraph(uint32_t engineId, const std::string& messageName, const std::shared_ptr<void>& dataPtr)
{
    std::shared_ptr<hiai::Graph> graph = hiai::Graph::GetInstance(1057997826);
    if (nullptr == graph)
    {
        HIAI_ENGINE_LOG("Fail to get the graph-%u", 1057997826);
        return -1;
    }
 
    hiai::EnginePortID engine_id;
    engine_id.graph_id = GRAPH_ID;
    engine_id.engine_id = engineId;
    engine_id.port_id = 0;
    graph->SendData(engine_id, messageName, dataPtr);

    return 0;
}

static void *ThreadFunction(void *param) 
{
    Channel* agent_channel = (Channel*)param; 
    if (agent_channel == nullptr) {
        HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                        "get agent channel to send failed.");
        return NULL;
    }

    while (1){
        // construct registered request Message and read message
        std::shared_ptr<FaceRegisterData> regData = std::make_shared<FaceRegisterData>();
        PresenterErrorCode agent_ret = agent_channel->ReceiveMessage(regData->response_rec);
        if (agent_ret != PresenterErrorCode::kNone){
            usleep(200000);
            continue;
        }

        HIAI_ENGINE_LOG("faceregister Receive face register req");
        int ret = SendDataByGraph(kFaceRegisterEngineId, "FaceRegisterData", std::static_pointer_cast<void>(regData));
        if (ret == HIAI_QUEUE_FULL) {        
            HIAI_ENGINE_LOG("Queue is full, sleep 200 ms");
            usleep(200);
        }
    }

    return NULL;
}

void StartFaceRegisterListen(Channel* agent_channel) 
{
    // 线程ID 
    pthread_t  ntid; 

    pthread_create(&ntid, NULL, &ThreadFunction, (void *)agent_channel);
    pthread_detach(ntid);
}
