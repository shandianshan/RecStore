set(RECSTORE_PATH ${PROJECT_SOURCE_DIR})
configure_file(${PROJECT_SOURCE_DIR}/src/recstore_config.h.in
	${PROJECT_BINARY_DIR}/recstore_config.h
)
include_directories(${PROJECT_BINARY_DIR})


add_subdirectory(base)

add_subdirectory(base_ps)
add_subdirectory(grpc_ps)
add_subdirectory(rdma_ps)

add_subdirectory(memory)
add_subdirectory(storage)
add_subdirectory(optimizer)

# add_subdirectory(kg)
add_subdirectory(test)

add_subdirectory(small_tests)


add_subdirectory(framework)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
	add_subdirectory(executable/uvm)
else()
	message(STATUS "Skipping executable/uvm because CUDA is disabled or nvcc not found")
endif()