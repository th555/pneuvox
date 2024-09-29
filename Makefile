CXX=g++

NAME = pneuvox
SRC_NOGUI_NO_MAIN = PV_Chamber.cpp PV_Pneunet.cpp PV_Valve.cpp PV_Controller.cpp PV_Sensor.cpp PV_Conduit.cpp Biquad.cpp
SRC_NOGUI = pneuvox.cpp $(SRC_NOGUI_NO_MAIN)
SRC_GUI =  PV_MeshRender.cpp
SRC = $(SRC_NOGUI) $(SRC_GUI)
OBJ = $(subst .cpp,.o,$(SRC))
OBJ_NOGUI = $(subst .cpp,.o,$(SRC_NOGUI))
SRC_TEST = test.cpp $(SRC_NOGUI_NO_MAIN)
OBJ_TEST = $(subst .cpp,.o,$(SRC_TEST))
LDFLAGS_NOGUI = -L../voxelyze/lib
LDFLAGS_GUI = -L../raylib/lib
LDFLAGS = $(LDFLAGS_NOGUI) $(LDFLAGS_GUI)
LDLIBS_NOGUI = -lvoxelyze.0.9 -lpthread
LDLIBS_GUI = -lraylib
LDLIBS = $(LDLIBS_NOGUI) $(LDLIBS_GUI)
INCLUDE_NOGUI = -I../voxelyze/include -isystem../tiny-dnn
INCLUDE_GUI = -I../raylib/include -I../raylib/src
INCLUDE = $(INCLUDE_NOGUI) $(INCLUDE_GUI)

CFLAGS = -c -Wall -O3

all: pneuvox

das6: CFLAGS += -march=znver2 -DNOGUI $(INCLUDE_NOGUI)
das6: $(OBJ_NOGUI)
	$(CXX) -o $(NAME) $(OBJ_NOGUI) $(LDFLAGS_NOGUI) $(LDLIBS_NOGUI)

debug: CFLAGS = -c -Wall -O0 -g $(INCLUDE)
debug: $(OBJ)
	$(CXX) -o $(NAME) $(OBJ) $(LDFLAGS) $(LDLIBS)

pneuvox: CFLAGS += $(INCLUDE)
pneuvox: $(OBJ)
	$(CXX) -o $(NAME) $(OBJ) $(LDFLAGS) $(LDLIBS)

test: CFLAGS += $(INCLUDE)
test: $(OBJ_TEST)
	$(CXX) -o test $(OBJ_TEST) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) $<

clean:
	rm -f $(OBJ)

fclean: clean
	rm -f $(NAME)
