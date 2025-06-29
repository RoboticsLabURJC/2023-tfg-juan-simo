cmake_minimum_required(VERSION 3.8)
project(f1_cars)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

include_directories(
  ${GAZEBO_INCLUDE_DIRS}
)


# the place where gazebo puts the models by default
set(GAZEBO_MODEL_PATH $ENV{HOME}/.gazebo/models)

add_custom_target(COMMAND cmake -E make_directory ${GAZEBO_MODEL_PATH})

install(DIRECTORY
  worlds
  launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  models/f1
  models/f1_astonMartin
  models/f1_dummy
  models/f1_guest
  models/f1_redBull
  models/f1_renault
  models/f1_renault_laser
  models/many_curves
  models/montmelo_line
  models/montreal_line
  models/nurburgring_line
  models/simple_circuit
  
  models/autocar
  models/f1_renault_ackerman
  models/f1_renault_camera
  models/big_simple_circuit
  models/big_montmelo_line
  models/big_montreal_line
  models/big_nurburgring_line
  models/big_many_curves
  DESTINATION ${GAZEBO_MODEL_PATH})

ament_package()


