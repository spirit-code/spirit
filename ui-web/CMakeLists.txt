MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- UI - Web ------------------------------------- <<" )

######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(SpiritWeb)
#############################################



######### Have the binary placed into the source head
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#############################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################




# ######### Tell CMake to create the executable
# add_library( SpinEngine STATIC
# 	${SOURCE_GROUP_DATA}
# 	${SOURCE_GROUP_ENGINE}
# 	${SOURCE_GROUP_UTILITY}
#     ${HEADER_GROUP_DATA}
#     ${HEADER_GROUP_ENGINE}
# 	${HEADER_GROUP_UTILITY} main_js.cxx)
# #############################################
# ######### Tell CMake to create the executable
# add_executable(spirit  main_js.cxx)
# # add_executable(spirit  ${SOURCE} ${UI_SOURCE} ${HEADERS} resources.qrc)
# #############################################


### Include Core headers
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${SPIRIT_INCLUDE_DIRS})

add_definitions( -DFMT_HEADER_ONLY )

######### Declare File groups ###############
### Header Gropus
set(HEADER_UI_WEB_ROOT)
### Source Groups
set(SOURCE_UI_WEB_ROOT)
#############################################


######### Add Subdirectories
# add_subdirectory(${PROJECT_SOURCE_DIR}/src)
# add_subdirectory(${PROJECT_SOURCE_DIR}/include)
#############################################



### Create Core Interface
# add_library( coreinterface STATIC main_js.cxx )

### Link
# target_link_libraries(coreinterface core)

# add_library( coreinterface STATIC
#     ${HEADER_CORE_DATA}
#     ${HEADER_CORE_ENGINE}
#     ${HEADER_CORE_UTILITY}
#     ${SOURCE_CORE_DATA}
#     ${SOURCE_CORE_ENGINE}
#     ${SOURCE_CORE_UTILITY}
#     main_js.cxx)


######## Set export functions for Web UI #############################
### Place all functions which should be exported by emcc into this list
set( INTERFACE_EXPORT_FUNCTIONS
    '_Spirit_Version' '_Spirit_Version_Revision' '_Spirit_Version_Full'
    # State
    '_State_Setup'
    # System
    '_System_Get_Index' '_System_Get_NOS' '_System_Get_Spin_Directions'
    # Chain
    '_Chain_Get_Index' '_Chain_Get_NOI'
    '_Chain_next_Image' '_Chain_prev_Image'
    '_Chain_Image_to_Clipboard' '_Chain_Insert_Image_Before' '_Chain_Insert_Image_After' '_Chain_Replace_Image' '_Chain_Delete_Image'
    '_Chain_Update_Data' '_Chain_Setup_Data'
    # Collection
    '_Collection_Get_NOC'
    '_Collection_next_Chain' '_Collection_prev_Chain'
    # Geometry
    '_Geometry_Get_Positions'
    '_Geometry_Get_Bounds' '_Geometry_Get_Center'
    '_Geometry_Get_Basis_Vectors' '_Geometry_Get_N_Cells' '_Geometry_Get_Translation_Vectors' '_Geometry_Get_Dimensionality'
    # Hamiltonian
    '_Hamiltonian_Set_Boundary_Conditions' '_Hamiltonian_Set_mu_s' '_Hamiltonian_Set_Field' '_Hamiltonian_Set_Exchange' '_Hamiltonian_Set_DMI' '_Hamiltonian_Set_Anisotropy'
    '_Hamiltonian_Get_Boundary_Conditions' '_Hamiltonian_Get_mu_s' '_Hamiltonian_Get_Field' '_Hamiltonian_Get_Exchange' '_Hamiltonian_Get_DMI' '_Hamiltonian_Get_Anisotropy'
    # Parameters LLG
    '_Parameters_Set_LLG_Time_Step' '_Parameters_Set_LLG_Damping'
    '_Parameters_Set_LLG_STT' '_Parameters_Set_LLG_Temperature'
    '_Parameters_Get_LLG_Time_Step' '_Parameters_Get_LLG_Damping'
    '_Parameters_Get_LLG_STT' '_Parameters_Get_LLG_Temperature'
    # Parameters GNEB
    '_Parameters_Set_GNEB_Spring_Constant' '_Parameters_Set_GNEB_Climbing_Falling'
    '_Parameters_Get_GNEB_Spring_Constant' '_Parameters_Get_GNEB_Climbing_Falling' '_Parameters_Get_GNEB_N_Energy_Interpolations'
    # Configurations
    '_Configuration_Domain' '_Configuration_PlusZ' '_Configuration_MinusZ'
    '_Configuration_Random' '_Configuration_Add_Noise_Temperature' '_Configuration_Skyrmion' '_Configuration_SpinSpiral'
    # Transitions
    '_Transition_Homogeneous' '_Transition_Add_Noise_Temperature'
    # Simulation
    '_JS_LLG_Iteration'
    '_Simulation_SingleShot' '_Simulation_PlayPause' '_Simulation_Stop_All'
    '_Simulation_Running_Image' '_Simulation_Running_Chain' '_Simulation_Running_Collection'
    '_Simulation_Running_Anywhere_Chain' '_Simulation_Running_Anywhere_Collection'
    # Log
    '_Log_Send' '_Log_Get_N_Entries' '_Log_Append' '_Log_Dump'
    )
### Replace ; in the list with , while transforming into a string
string( REPLACE ";" ", " INTERFACE_EXPORT_FUNCTIONS_STRING "${INTERFACE_EXPORT_FUNCTIONS}")
######################################################################



# ### Custom command to transpile the interface to JS using emscripten
# add_custom_command(
#     OUTPUT core.js
#     COMMAND emcc ${CMAKE_BINARY_DIR}/libcoreinterface.a -o ${PROJECT_SOURCE_DIR}/core.js -s EXPORTED_FUNCTIONS="['_test']" -O2 -s ALLOW_MEMORY_GROWTH=1
#     WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
# )

### Custom target to create dependency on core.js, so that the custom command is executed
# add_custom_target(corejs ALL DEPENDS core.js coreinterface)


add_executable( spirit_js main_js.cxx
                ${HEADER_UI_WEB_ROOT}
                ${SOURCE_UI_WEB_ROOT}
                ${SPIRIT_OBJS} )

# target_link_libraries( spirit_js ${SPIRIT_LIBRARIES_STATIC} )

set_target_properties( spirit_js PROPERTIES OUTPUT_NAME "spirit" )
### We want it to be placed under ui-web/ s.t. it is directly part of the python core module/package
set( SPIRIT_JS_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/ui-web/ )
set_target_properties( spirit_js PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${SPIRIT_JS_OUTPUT_DIR} )

### Set specific flags for emcc
set_property(TARGET spirit_js PROPERTY COMPILE_FLAGS "-s DISABLE_EXCEPTION_CATCHING=2 -s ASSERTIONS=1 -s WASM=1" )
set_property(TARGET spirit_js PROPERTY LINK_FLAGS    "-s ALLOW_MEMORY_GROWTH=1 -s WASM=1 -s EXPORTED_FUNCTIONS=\"[${INTERFACE_EXPORT_FUNCTIONS_STRING}]\"" )


### Non-transpile version
# add_executable(corejs main_js.cxx)
# target_link_libraries(corejs core)


######### Header and Source messages ########
if ( PRINT_SOURCES )
    MESSAGE( STATUS ">> Headers:                    ${HEADER_UI_WEB_ROOT}" )
    MESSAGE( STATUS ">> Sources:                    ${SOURCE_UI_WEB_ROOT}" )
endif ()
#############################################


MESSAGE( STATUS ">> --------------------- UI - Web done -------------------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )