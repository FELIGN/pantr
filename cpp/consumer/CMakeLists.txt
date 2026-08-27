# A consumer of the INSTALLED pantr, deliberately NOT part of pantr's own build.
#
# This directory is never added by the top-level CMakeLists. It is configured
# standalone against an install prefix, which is the only way to exercise what a
# third party actually gets: the top-level build reaches the headers through
# BUILD_INTERFACE and the in-tree `pantr::core` alias, so it cannot tell whether
# the installed package works. The first run of this found that the exported
# target was named `pantr::pantr_core`.
#
# scripts/ci_local.sh's `consumer` section drives it. By hand:
#
#     cmake -S cpp/consumer -B /tmp/c -DCMAKE_PREFIX_PATH=<prefix> && cmake --build /tmp/c
cmake_minimum_required(VERSION 3.25)
project(pantr_consumer LANGUAGES CXX)

find_package(pantr REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE pantr::core)
target_compile_features(consumer PRIVATE cxx_std_20)
