cmake_minimum_required(VERSION 3.10)  # 推荐使用更新的版本

project(myanns)

set(CMAKE_CXX_STANDARD 11)  # 设置 C++ 标准
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -O3")

# 搜索所有 .cpp 文件
file(GLOB SOURCES *.cpp)

# 为每个文件创建一个可执行目标
foreach(SOURCE ${SOURCES})
    # 获取文件名（无扩展名）
    get_filename_component(TARGET_NAME ${SOURCE} NAME_WE)
    # 添加可执行文件
    add_executable(${TARGET_NAME} ${SOURCE})
    # 链接库，如果有其他库，请在这里添加
    target_link_libraries(${TARGET_NAME} -ltcmalloc)
endforeach()
