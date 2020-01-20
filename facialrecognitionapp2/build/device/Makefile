SHELL := /bin/bash
all : libDevice.so
#DEVICE COMPILER		
DDK_HOME = /home/syw/tools/che/ddk/ddk
CC := aarch64-linux-gnu-g++
#Attention: no space at the end of the line, or it will be added to dir name which will will cause error
LOCAL_DIR  :=  ../../device_engines ../../common

SRC_DIR = $(LOCAL_DIR)
BUILD_DIR = tmp
OUT_DIR = ../../out
OBJ_DIR = $(BUILD_DIR)/obj
DEPS_DIR  = $(BUILD_DIR)/deps

#Add other .h file paths here
INC_DIR = \
    -I$(DDK_HOME)/include/inc \
    -I$(DDK_HOME)/include/inc/custom \
    -I$(DDK_HOME)/include/third_party/opencv/include \
    -I$(DDK_HOME)/include/third_party/protobuf/include \
    -I$(DDK_HOME)/include/third_party/cereal/include \
    -I$(DDK_HOME)/include/libc_sec/include \
    -I../../device_engines \
    -I../../common/include 

#Add compile arguments here 
CC_FLAGS := $(INC_DIR) -g -std=c++11 -fPIC -DCPU_ONLY
LNK_FLAGS := \
    -L$(DDK_HOME)/device/lib/ \
    -lDvpp_api \
    -lDvpp_jpeg_decoder \
    -lDvpp_jpeg_encoder \
    -lDvpp_vpc \
    -lmedia_mini \
    -lhiai_server \
    -lidedaemon \
    -lhiai_common \
    -lopencv_world \
    -shared

#Recursivly traverse 3rd level subdirectory here
DIRS := $(shell find $(SRC_DIR) -maxdepth 3 -type d)
CUSTOM_DIRS := $(shell find $(SRC_DIR) -maxdepth 3 -type d)

#Add each subdirectory to search path
VPATH = $(DIRS)

#Search all cpp files in src_dir and according subdirectories
SOURCES  = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
CUSTOM_SOURCES  = $(foreach dir, $(CUSTOM_DIRS), $(wildcard $(dir)/*.cpp))
OBJS   = $(addprefix $(OBJ_DIR)/,$(patsubst %.cpp,%.o,$(notdir $(SOURCES))))
OBJS_customop = $(addprefix $(OBJ_DIR)/,$(patsubst %.cpp,%.o,$(notdir $(CUSTOM_SOURCES))))
OBJS_no_customop := $(filter-out $(OBJS_customop), $(OBJS))
DEPS  = $(addprefix $(DEPS_DIR)/, $(patsubst %.cpp,%.d,$(notdir $(SOURCES))))

#Compile source code and link to target so

libDevice.so: $(OBJS_customop)
	$(CC) $^ $(LNK_FLAGS) -o $@

#Create OBJ path before compiling to make sure path exists
$(OBJ_DIR)/%.o:%.cpp %.h
	@if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi;
	$(CC) -c $(CC_FLAGS) -o $@ $<


#Create DEPS path before compiling to make sure path exists
#@ means execute print of hidden command
$(DEPS_DIR)/%.d:%.cpp %.h
	@if [ ! -d $(DEPS_DIR) ]; then mkdir -p $(DEPS_DIR); fi;
	set -e; rm -f $@;
	$(CC) -MM $(CC_FLAGS) $< > $@.$$$$;
	sed 's,\($*\)\.o[ :]*,$(OBJ_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@;
	rm -f $@.$$$$

#Adding "-" in the head means ignoring errors 
ifneq ($(MAKECMDGOALS), clean)
	-include $(DEPS)
endif

.PHONY : clean install
clean:
	rm -rf $(BUILD_DIR) lib*.so *.o

install: libDevice.so
	cp *.so $(OUT_DIR)
