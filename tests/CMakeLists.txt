# tests/CMakeLists.txt
add_executable(test_ocr test_ocr.cpp)
target_link_libraries(test_ocr PRIVATE libocr Catch2::Catch2WithMain)  # 链接共享库 + Catch2

# 添加测试
add_test(NAME TestOCR COMMAND test_ocr)