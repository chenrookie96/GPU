set(CMAKE_CXX_COMPILER "${MUSA_PATH}/bin/mcc" CACHE PATH "Using mcc" FORCE)

set(MUSA_MCC_FLAGS "${MUSA_MCC_FLAGS}"
  "SHELL:-mllvm -mtgpu-assume-nested-constant-as-global"
  "SHELL:-mllvm -misched-topdown"
  "SHELL:-mllvm -mtgpu-complex-promote-alloca-opt=1"
  "SHELL:-mllvm -mtgpu-promote-alloca-to-vector-limit=512"
  "SHELL:-mllvm -mtgpu-promote-alloca-to-vector-num-reg-limit=128"
  "-ffast-math"
)

message("MUSA_MCC_FLAGS: ${MUSA_MCC_FLAGS}")
function(add_deep_ep_library target_name source_file)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CXX)
    add_library(${target_name} STATIC ${source_file})
    set_target_properties(${target_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD_REQUIRED ON
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF

    )
    target_compile_options(${target_name} PRIVATE ${MUSA_MCC_FLAGS})
    target_link_options(${target_name} PUBLIC -rdc=true --offload-arch=${MUSA_ARCHS})
    target_link_libraries(${target_name} PUBLIC nvshmem_host nvshmem_device musart mlx5)
endfunction()

add_subdirectory(legacy)
add_subdirectory(backend)
