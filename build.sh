#!/bin/sh
export LD_LIBRARY_PATH=/home/pass/tools/che/ddk/ddk/uihost/lib/
TOP_DIR=${PWD}
BUILD_DIR=${TOP_DIR}/build
OUTPUT_DIR=${TOP_DIR}/out

if [ ! -d ${BUILD_DIR} ];then
  mkdir -p ${BUILD_DIR}
fi

# execute cmake command
cd ${BUILD_DIR}
cmake .. && make -j8
if [ $? != 0 ];then
  rm -rf ${BUILD_DIR}/*
  echo "============ cmake project failed."
  exit
fi

# Check the graph.config file in your work directory.
cd ${TOP_DIR}
if [ ! -f "graph.config" ];then
  echo "============ graph.config file not exist in ${TOP_DIR}! ============"
  exit
fi
cp  -f ${TOP_DIR}/graph.config ${OUTPUT_DIR}/graph.config 

# Delete temporary directory or files in end.
rm -rf ${BUILD_DIR}
if [ $? != 0 ];then
  echo "============ rm -rf ${BUILD_DIR} failed."
  exit
fi

echo "============ build success ! ============"

scp -r out/ HwHiAiUser@172.16.117.103:/home/HwHiAiUser/HIAI_PROJECTS/ascend_workspace/robot_dog/
