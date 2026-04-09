SRC_DIR = src
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.cpp) \
       $(wildcard $(SRC_DIR)/backend/*.cpp) \
       $(wildcard $(SRC_DIR)/model/*.cpp) \
       $(wildcard $(SRC_DIR)/infer/*.cpp)

OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

muduo: $(OBJS)
	g++  -g -Wall -Wextra $(OBJS) -O3 -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	g++ -O3 -I. -MMD -MP -c $< -o $@

-include $(OBJS:.o=.d)

clean:
	rm -rf $(BUILD_DIR) muduo
