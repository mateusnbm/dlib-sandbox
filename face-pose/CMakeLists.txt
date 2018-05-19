#
#
#

# Macros.

macro(add_example name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib )
endmacro()

macro(add_gui_example name)
   if (DLIB_NO_GUI_SUPPORT)
      message("No GUI support, so we won't build the ${name} example.")
   else()
      add_example(${name})
   endif()
endmacro()

macro(add_opencv_example name)
   if (DLIB_NO_GUI_SUPPORT)
      message("No GUI support, so we won't build the ${name} example.")
   else()
      find_package(OpenCV QUIET)
      if (OpenCV_FOUND)
         include_directories(${OpenCV_INCLUDE_DIRS})
         add_executable(${name} ${name}.cpp)
         target_link_libraries(${name} dlib::dlib ${OpenCV_LIBS} )
      else()
         message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
      endif()
   endif()
endmacro()

# Main.

cmake_minimum_required(VERSION 2.8.12)

project(examples)

add_subdirectory(../dlib-19.10 dlib_build)

add_opencv_example(face-pose-example)
