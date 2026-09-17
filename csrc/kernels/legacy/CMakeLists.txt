set_source_files_properties(internode_ll.cu PROPERTIES
  COMPILE_OPTIONS "-Od3;-mllvm;-mtgpu-normal-access-GV=1"
)

add_deep_ep_library(legacy_layout_cuda layout.cu)
add_deep_ep_library(legacy_intranode_cuda intranode.cu)
add_deep_ep_library(legacy_internode_cuda internode.cu)
add_deep_ep_library(legacy_internode_ll_cuda internode_ll.cu)
add_deep_ep_library(legacy_intranode_ace_cuda intranode_ace.cu)

# Link these libraries later
set(LEGACY_CUDA_LIBRARIES legacy_layout_cuda legacy_intranode_cuda legacy_internode_cuda legacy_internode_ll_cuda legacy_intranode_ace_cuda CACHE INTERNAL "Legacy kernels")
