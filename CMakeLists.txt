cmake_minimum_required(VERSION 3.20)
project(PaddleOCRONNXService VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg 工具链
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Git 版本信息
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_VERSION)
    set(GIT_VERSION "unknown")
endif()
message(STATUS "Git Version: ${GIT_VERSION}")

# 构建时间（格式 YYYY-MM-DD HH:MM:SS）
string(TIMESTAMP BUILD_TIME "%Y-%m-%d %H:%M:%S")
message(STATUS "Build Time: ${BUILD_TIME}")

# 测试选项（默认 ON）
option(BUILD_TESTS "Build unit tests" ON)
message(STATUS "Build Tests: ${BUILD_TESTS}")

# 查找包
find_package(OpenCV REQUIRED)
find_package(unofficial-onnxruntime CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
if(BUILD_TESTS)
    find_package(Catch2 CONFIG REQUIRED)  # 仅测试时查找
endif()

# 共享库：提取 src/ 核心代码（避免重复编译）
add_library(libocr STATIC
    src/ocr_detect.cpp
    src/ocr_recognize.cpp
    src/ocr_inference.cpp
    src/ocr_service.cpp
)
target_include_directories(libocr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(libocr PRIVATE
    OpenCV::opencv_world
    unofficial::onnxruntime::onnxruntime
    spdlog::spdlog
)
target_compile_definitions(libocr PRIVATE GIT_VERSION="${GIT_VERSION}" BUILD_TIME="${BUILD_TIME}")

# 主 exe：链接 libocr + main
add_executable(ocr_server src/main.cpp)
target_link_libraries(ocr_server PRIVATE libocr)

# Windows DLL 复制（针对主 exe）
if(WIN32)
    add_custom_command(TARGET ocr_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:ocr_server>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:unofficial::onnxruntime::onnxruntime>
        $<TARGET_FILE:OpenCV::opencv_world>
        $<TARGET_FILE_DIR:ocr_server>
    )
endif()

# 安装（主 exe）
install(TARGETS ocr_server DESTINATION bin)

# 测试（可选）
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()