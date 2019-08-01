### Makefile for building tensorflow application

SOURCE_DIR =./src
EXE_FILE = predict

CPP     = g++ -std=c++11 -Wl,--no-as-needed
LDFLAGS = -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w -Wl,--allow-multiple-definition -Wl,--whole-archive
LDLIBS  = -L./lib 
INCLUDES = -I./include -I./include/eigen3 -I./include/google -I./include/nsync
#INCLUDES = -I./include/eigen3 -I./include/google -I./include/nsync -I./include

ACTUAL_LIBS = -ldl -lz -lpthread -ltensorflow_cc -ltensorflow_framework

#INPUT_FILE = $(SOURCE_DIR)/main.cpp $(SOURCE_DIR)/fm_model_loader.cpp libtensorflow-core.a nsync.a libprotobuf.a libprotoc.a
#INPUT_FILE = $(SOURCE_DIR)/main.cpp $(SOURCE_DIR)/fm_model_loader.cpp lib/nsync.a lib/libprotobuf.a lib/libprotoc.a
#INPUT_FILE = client.cc
#INPUT_FILE = main.cc
INPUT_FILE = ./src/predict.cc

$(EXE_FILE):${INPUT_FILE}
	$(CPP) -o $(EXE_FILE) $(INCLUDES) $(LDFLAGS) $(LDLIBS) $(ACTUAL_LIBS) $(INPUT_FILE)
clean:
	rm -rf ${EXE_FILE}
