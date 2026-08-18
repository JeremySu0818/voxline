cmake_minimum_required(VERSION 3.26)
project(voxline_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(VOXLINE_BUILD_HOST_NEMOTRON_TRACE "Build host-side Nemotron runtime trace executable" OFF)

include(FetchContent)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

FetchContent_Declare(
    opencl_headers
    URL https://github.com/KhronosGroup/OpenCL-Headers/archive/8a97ebc88daa3495d6f57ec10bb515224400186f.tar.gz
    URL_HASH SHA256=211b3897f91136a9d4ceeb7b892cae88388408ff6c4d06428e9d3f0face2ee53
)
FetchContent_Declare(
    sentencepiece_source
    URL https://github.com/google/sentencepiece/archive/31646a467d2051eb904e0b45de3a73e91fe1c1e3.tar.gz
    URL_HASH SHA256=b4eb17e6bea5c9380ddd79d04d01d6aa5e280be8f3e3cef1f745a80e6d9451ce
)
FetchContent_Declare(
    ggml_source
    URL https://github.com/ggml-org/ggml/archive/c03b4e2bcece5134827881af90242086daf75be5.tar.gz
    URL_HASH SHA256=7f1fe40bdd9d75bc675dab06c0436308a69556b71125bd1c9c668b26a641d94c
    SOURCE_SUBDIR cmake/voxline-no-add
)
FetchContent_Declare(
    nemo_speech_source
    URL https://github.com/NVIDIA/NeMo-Speech.cpp/archive/9bc876635af36df537d9bc6d3f57ad1b76e4f74a.tar.gz
    URL_HASH SHA256=3970d62d9d5c1fb3370253d793d4a19ba201d84f013171ec15f5e6b08981a1df
    SOURCE_SUBDIR cmake/voxline-no-add
)

set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sentencepiece_source)
set(SENTENCEPIECE_DIR "${sentencepiece_source_SOURCE_DIR}")

set(NEMO_SPEECH_BUILD_ASR ON CACHE BOOL "" FORCE)
set(NEMO_SPEECH_BUILD_DIAR OFF CACHE BOOL "" FORCE)
set(NEMO_SPEECH_BUILD_TTS OFF CACHE BOOL "" FORCE)
set(NEMO_SPEECH_BUILD_NMT OFF CACHE BOOL "" FORCE)
set(NEMO_SPEECH_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(NEMO_SPEECH_BUILD_HTTP OFF CACHE BOOL "" FORCE)
set(NEMO_SPEECH_BUILD_GRPC OFF CACHE BOOL "" FORCE)
set(NEMO_SPEECH_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NEMO_SPEECH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEMO_SPEECH_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(NEMO_SPEECH_GGML_PATCHED OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
if(ANDROID)
    FetchContent_MakeAvailable(opencl_headers)
    set(VOXLINE_OPENCL_HEADERS_DIR "${opencl_headers_SOURCE_DIR}" CACHE INTERNAL "")

    add_library(voxline_opencl_link_stub SHARED opencl_link_stub.c)
    set_target_properties(
        voxline_opencl_link_stub
        PROPERTIES
        OUTPUT_NAME OpenCL
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/opencl-link-stub"
    )

    # Build ggml's Android CPU variants as loadable modules. The runtime picks
    # the fastest backend supported by the device instead of being stuck on
    # baseline ARMv8 Q8 kernels or risking SIGILL from a hard-coded -march.
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    set(GGML_OPENCL ON CACHE BOOL "" FORCE)
    set(GGML_OPENCL_USE_ADRENO_KERNELS OFF CACHE BOOL "" FORCE)
    # ggml Vulkan is currently unsafe on Android/Adreno in this runtime. We have
    # observed reproducible driver SIGSEGVs from both cache-aware encoder and
    # fused RNNT dispatch. Keep Android on the feature-dispatched ARM64 CPU
    # backends until the Vulkan path has an isolated device-level correctness
    # and stability test.
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
endif()
# The downloaded archives are not Git worktrees. Prevent ggml from walking up
# into the app repository and embedding the app's commit/dirty state.
set(GIT_EXE GIT_EXE-NOTFOUND CACHE FILEPATH "" FORCE)

set(SENTENCEPIECE_STATIC_LIB sentencepiece-static CACHE STRING "" FORCE)
set(SENTENCEPIECE_INCLUDE_DIR "${SENTENCEPIECE_DIR}/src" CACHE PATH "" FORCE)

FetchContent_MakeAvailable(ggml_source nemo_speech_source)

if(ANDROID AND GGML_OPENCL)
    # ggml OpenCL currently leaves the optional tensor-memset hook unset, but
    # NeMo RNNT persistent decoder state requires it during model setup. Fill
    # the backend capability with the standard OpenCL buffer-fill operation.
    # Keep this patch device-agnostic: no vendor, GPU family, or model check.
    set(_GGML_OPENCL_CPP "${ggml_source_SOURCE_DIR}/src/ggml-opencl/ggml-opencl.cpp")
    file(READ "${_GGML_OPENCL_CPP}" _GGML_OPENCL_TEXT)
    if(NOT _GGML_OPENCL_TEXT MATCHES "ggml_backend_opencl_buffer_memset_tensor")
        string(REPLACE
            "static void ggml_backend_opencl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {"
            "static void ggml_backend_opencl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {\n    GGML_ASSERT(tensor->extra);\n    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer->buft->device);\n    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;\n    sync_with_other_backends(backend_ctx);\n    CL_CHECK(clEnqueueFillBuffer(\n        backend_ctx->queue, extra->data_device, &value, sizeof(value),\n        extra->offset + tensor->view_offs + offset, size, 0, NULL, NULL));\n    CL_CHECK(clFinish(backend_ctx->queue));\n}\n\nstatic void ggml_backend_opencl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {"
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
        string(REPLACE
            "/* .memset_tensor   = */ NULL,"
            "/* .memset_tensor   = */ ggml_backend_opencl_buffer_memset_tensor,"
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
        file(WRITE "${_GGML_OPENCL_CPP}" "${_GGML_OPENCL_TEXT}")
    endif()

    # Non-quantized tensor uploads must include a view's byte offset. Without
    # it, cached/view inputs overwrite the base tensor instead of the selected
    # slice.
    if(_GGML_OPENCL_TEXT MATCHES
       "queue, extra->data_device, CL_TRUE, extra->offset \\+ offset,")
        string(REPLACE
            "queue, extra->data_device, CL_TRUE, extra->offset + offset,"
            "queue, extra->data_device, CL_TRUE, extra->offset + tensor->view_offs + offset,"
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
    endif()

    # Q8 SOA sub-buffers are rejected by some public OpenCL runtimes even when
    # the requested region satisfies ggml's alignment. The kernels only need
    # two valid cl_mem objects, so allocate the scale and quant planes directly.
    if(NOT _GGML_OPENCL_TEXT MATCHES "Q8 SOA components are independent buffers")
        string(REPLACE
            [=[        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);]=]
            [=[        // Q8 SOA components are independent buffers. Some OpenCL implementations
        // reject aligned sub-regions of ggml's packed parent allocation, while
        // the conversion/matmul kernels only require valid cl_mem objects.
        extra->d = clCreateBuffer(context, CL_MEM_READ_WRITE, size_d, NULL, &err);
        CL_CHECK(err);
        extra->q = clCreateBuffer(context, CL_MEM_READ_WRITE, size_q, NULL, &err);
        CL_CHECK(err);]=]
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
    endif()

    # The OpenCL Q8 kernels only implement contiguous operand layouts. Reject
    # permuted/batched views so the scheduler routes them to a correct backend.
    if(_GGML_OPENCL_TEXT MATCHES
       "op->src\[0\]->type == GGML_TYPE_Q8_0\\) \\{\n                return op->src\[1\]->type == GGML_TYPE_F32;")
        string(REPLACE
            [=[            } else if (op->src[0]->type == GGML_TYPE_Q8_0) {
                return op->src[1]->type == GGML_TYPE_F32;
            }]=]
            [=[            } else if (op->src[0]->type == GGML_TYPE_Q8_0) {
                return op->src[1]->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(op->src[0]) &&
                       ggml_is_contiguous(op->src[1]);
            }]=]
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
    endif()
    file(WRITE "${_GGML_OPENCL_CPP}" "${_GGML_OPENCL_TEXT}")

endif()

if(ANDROID AND GGML_OPENCL)
    # Keep RNNT argmax on the accelerator. Falling this final reduction back
    # to CPU creates a split at the decoder output and, on OpenCL, returned
    # stale token ids even when the device logits were correct. This portable
    # scalar-per-row kernel is intentionally simple: RNNT has only a handful
    # of rows per call, while each row is the vocabulary dimension.
    set(_GGML_OPENCL_CPP "${ggml_source_SOURCE_DIR}/src/ggml-opencl/ggml-opencl.cpp")
    file(READ "${_GGML_OPENCL_CPP}" _GGML_OPENCL_TEXT)
    if(NOT _GGML_OPENCL_TEXT MATCHES "program_argmax_f32_i32")
        string(REPLACE
            "    cl_program program_softmax_4_f16;\n    cl_program program_argsort_f32_i32;"
            "    cl_program program_softmax_4_f16;\n    cl_program program_argmax_f32_i32;\n    cl_program program_argsort_f32_i32;"
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
        string(REPLACE
            "    cl_kernel kernel_soft_max_f16, kernel_soft_max_4_f16;\n    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f16;"
            "    cl_kernel kernel_soft_max_f16, kernel_soft_max_4_f16;\n    cl_kernel kernel_argmax_f32_i32;\n    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f16;"
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
        string(REPLACE
            "    // argsort\n    {"
            [=[    // argmax
    {
        const std::string kernel_src = R"CLC(
kernel void kernel_argmax_f32_i32(
        global const uchar * src0_base,
        ulong offset0,
        global uchar * dst_base,
        ulong offsetd,
        int ne00,
        ulong nb01,
        int nrows) {
    const int row = (int)get_global_id(0);
    if (row >= nrows || ne00 <= 0) return;
    global const float * values =
        (global const float *)(src0_base + offset0 + (ulong)row * nb01);
    float best = values[0];
    int best_index = 0;
    for (int col = 1; col < ne00; ++col) {
        const float value = values[col];
        if (value > best) {
            best = value;
            best_index = col;
        }
    }
    ((global int *)(dst_base + offsetd))[row] = best_index;
}
)CLC";
        backend_ctx->program_argmax_f32_i32 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->kernel_argmax_f32_i32 = clCreateKernel(
            backend_ctx->program_argmax_f32_i32, "kernel_argmax_f32_i32", &err), err));
        GGML_LOG_CONT(".");
    }

    // argsort
    {]=]
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
        string(REPLACE
            "        case GGML_OP_ARGSORT: {"
            "        case GGML_OP_ARGMAX:\n            return op->src[0]->type == GGML_TYPE_F32 &&\n                   op->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[0]) &&\n                   ggml_is_contiguous(op);\n        case GGML_OP_ARGSORT: {"
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
        string(REPLACE
            "static void ggml_cl_argsort(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {"
            [=[static void ggml_cl_argmax(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0 && src0->extra);
    GGML_ASSERT(dst && dst->extra);
    GGML_UNUSED(src1);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    auto * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    auto * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    auto * extrad = (ggml_tensor_extra_cl *) dst->extra;
    const cl_ulong offset0 = extra0->offset + src0->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;
    const int ne00 = src0->ne[0];
    const cl_ulong nb01 = src0->nb[1];
    const int nrows = ggml_nrows(src0);
    cl_kernel kernel = backend_ctx->kernel_argmax_f32_i32;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne00));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &nrows));

    size_t global_work_size[] = {(size_t)nrows, 1, 1};
    backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, nullptr, dst);
}

static void ggml_cl_argsort(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {]=]
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
        string(REPLACE
            "        case GGML_OP_ARGSORT:\n            if (!any_on_device) {"
            "        case GGML_OP_ARGMAX:\n            if (!any_on_device) {\n                return false;\n            }\n            func = ggml_cl_argmax;\n            break;\n        case GGML_OP_ARGSORT:\n            if (!any_on_device) {"
            _GGML_OPENCL_TEXT
            "${_GGML_OPENCL_TEXT}"
        )
        file(WRITE "${_GGML_OPENCL_CPP}" "${_GGML_OPENCL_TEXT}")
    endif()
endif()

if(ANDROID)
    # Session graph metadata is cached, but unnamed intermediates are placed in
    # one scheduler-owned liveness pool shared by every cached graph. RNNT
    # alternates joint and predictor graphs; retaining the previous graph's
    # buffer/data/extra pointers therefore makes the next cache hit read memory
    # that the other graph has reused. Remember tensors initially owned by the
    # scheduler and clear their placement before each allocation.
    set(_NEMO_RUNTIME_H
        "${nemo_speech_source_SOURCE_DIR}/src/runtime/ggml/runtime.h")
    set(_NEMO_SESSION_CPP
        "${nemo_speech_source_SOURCE_DIR}/src/runtime/ggml/session.cpp")
    file(READ "${_NEMO_RUNTIME_H}" _NEMO_RUNTIME_H_TEXT)
    file(READ "${_NEMO_SESSION_CPP}" _NEMO_SESSION_TEXT)
    if(NOT _NEMO_RUNTIME_H_TEXT MATCHES "scheduler_tensors")
        string(REPLACE
            "        std::vector<NodeAssignment> schedule;\n        // Cleared before allocation so views bind to the current SessionState."
            "        std::vector<NodeAssignment> schedule;\n        // Storage for these tensors belongs to the shared scheduler liveness pool.\n        std::vector<ggml_tensor*> scheduler_tensors;\n        // Cleared before allocation so views bind to the current SessionState."
            _NEMO_RUNTIME_H_TEXT
            "${_NEMO_RUNTIME_H_TEXT}"
        )
        string(REPLACE
            [=[        cr.gf = build_graph_into(
            root_module, cr.input_tensors, cr.output_tensors, sched_meta, cr.sched_meta);
        collect_state_views(cr.gf, cr.state_views);]=]
            [=[        cr.gf = build_graph_into(
            root_module, cr.input_tensors, cr.output_tensors, sched_meta, cr.sched_meta);
        {
            std::unordered_set<ggml_tensor*> seen;
            auto remember = [&](ggml_tensor* t) {
                if (t != nullptr && t->buffer == nullptr && seen.insert(t).second) {
                    cr.scheduler_tensors.push_back(t);
                }
            };
            const int n_nodes = ggml_graph_n_nodes(cr.gf);
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor* node = ggml_graph_node(cr.gf, i);
                remember(node);
                for (int j = 0; j < GGML_MAX_SRC; ++j) remember(node->src[j]);
            }
        }
        collect_state_views(cr.gf, cr.state_views);]=]
            _NEMO_SESSION_TEXT
            "${_NEMO_SESSION_TEXT}"
        )
        string(REPLACE
            "        const bool direct = !fresh && cr.direct_ok && state == nullptr && cr.state_views.empty();"
            "        const bool direct = !fresh && cr.direct_ok && state == nullptr &&\n                            cr.state_views.empty() && cr.scheduler_tensors.empty();"
            _NEMO_SESSION_TEXT
            "${_NEMO_SESSION_TEXT}"
        )
        string(REPLACE
            "        // Clear state views so allocation binds them to the current SessionState."
            "        // Rebind scheduler-owned intermediates; another cached graph may have reused the pool.\n        for (ggml_tensor* t : cr.scheduler_tensors) {\n            t->buffer = nullptr;\n            t->data = nullptr;\n            t->extra = nullptr;\n        }\n\n        // Clear state views so allocation binds them to the current SessionState."
            _NEMO_SESSION_TEXT
            "${_NEMO_SESSION_TEXT}"
        )
        file(WRITE "${_NEMO_RUNTIME_H}" "${_NEMO_RUNTIME_H_TEXT}")
        file(WRITE "${_NEMO_SESSION_CPP}" "${_NEMO_SESSION_TEXT}")
    endif()
endif()

if(ANDROID)
    # OpenCL's quantized SOA upload path currently assumes set_tensor receives
    # a complete tensor. The fused attention projection is assembled from
    # three GGUF tensors, so gather those bytes on the host and perform one
    # backend upload. Besides avoiding repeated conversion/allocation, this
    # honors set_tensor semantics on every backend without device checks.
    set(_NEMO_REL_POS_CPP
        "${nemo_speech_source_SOURCE_DIR}/src/asr/encoder/rel_pos_attention.cpp")
    file(READ "${_NEMO_REL_POS_CPP}" _NEMO_REL_POS_TEXT)
    if(NOT _NEMO_REL_POS_TEXT MATCHES "fused_qkv_data")
        string(REPLACE
            [=[        const size_t part_qs = (size_t)n_feat * n_feat;
        const size_t part_d = third - part_qs;
        const size_t total_qs = 3 * part_qs;
        const char* parts[3] = {".linear_q.weight", ".linear_k.weight", ".linear_v.weight"};
        for (int i = 0; i < 3; i++) {
]=]
            [=[        const size_t part_qs = (size_t)n_feat * n_feat;
        const size_t part_d = third - part_qs;
        const size_t total_qs = 3 * part_qs;
        std::vector<char> fused_qkv_data(ggml_nbytes(w.tensor));
        const char* parts[3] = {".linear_q.weight", ".linear_k.weight", ".linear_v.weight"};
        for (int i = 0; i < 3; i++) {
]=]
            _NEMO_REL_POS_TEXT
            "${_NEMO_REL_POS_TEXT}"
        )
        string(REPLACE
            [=[            if (planar_q8) {
                // Each source tensor is [qs_part][d_part]. Assemble the fused
                // allocation as [qs_q|qs_k|qs_v][d_q|d_k|d_v].
                ggml_backend_tensor_set(w.tensor, data, i * part_qs, part_qs);
                ggml_backend_tensor_set(w.tensor, data + part_qs, total_qs + i * part_d, part_d);
            } else {
                ggml_backend_tensor_set(w.tensor, data, i * third, third);
            }
        }
        if (planar_q8) {
]=]
            [=[            if (planar_q8) {
                // Each source tensor is [qs_part][d_part]. Assemble the fused
                // allocation as [qs_q|qs_k|qs_v][d_q|d_k|d_v].
                std::memcpy(fused_qkv_data.data() + i * part_qs, data, part_qs);
                std::memcpy(
                    fused_qkv_data.data() + total_qs + i * part_d,
                    data + part_qs,
                    part_d);
            } else {
                std::memcpy(fused_qkv_data.data() + i * third, data, third);
            }
        }
        ggml_backend_tensor_set(
            w.tensor, fused_qkv_data.data(), 0, fused_qkv_data.size());
        if (planar_q8) {
]=]
            _NEMO_REL_POS_TEXT
            "${_NEMO_REL_POS_TEXT}"
        )
        file(WRITE "${_NEMO_REL_POS_CPP}" "${_NEMO_REL_POS_TEXT}")
    endif()
endif()

if(ANDROID AND VOXLINE_NEMOTRON_NUMERICAL_TRACE)
    # Temporary numerical checkpoints for isolating accelerator correctness in
    # the first cache-aware FastConformer block. The checkpoint stores are graph
    # roots, so they observe the exact backend-resident tensors rather than a
    # rebuilt host approximation. They are read once by Voxline after the first
    # streaming chunk and will be removed once the OpenCL divergence is fixed.
    set(_NEMO_FASTCONFORMER_CPP
        "${nemo_speech_source_SOURCE_DIR}/src/asr/encoder/fastconformer.cpp")
    file(READ "${_NEMO_FASTCONFORMER_CPP}" _NEMO_FASTCONFORMER_TEXT)
    # Android's CPU and OpenCL PAD implementations both support asymmetric
    # left padding. Use it directly so a cache-aware encoder graph remains on
    # its selected accelerator; the portable PAD+ROLL compatibility path is
    # still retained for platforms whose PAD backend rejects left padding.
    if(NOT _NEMO_FASTCONFORMER_TEXT MATCHES
       "NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS.*defined\\(__ANDROID__\\)")
        string(REPLACE
            "#ifdef NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS\n    return ggml_pad_ext"
            "#if defined(NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS) || defined(__ANDROID__)\n    return ggml_pad_ext"
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        file(WRITE "${_NEMO_FASTCONFORMER_CPP}" "${_NEMO_FASTCONFORMER_TEXT}")
    endif()
    if(NOT _NEMO_FASTCONFORMER_TEXT MATCHES "encoder.diag.sub.pad0.store")
        string(REPLACE
            [=[        for (int l = 0; l < cfg_.n_layers; l++) {
            const std::string src =
                name_ + ".layers." + std::to_string(l) + ".conv.depthwise_conv.weight";
            session->model_tensor_container->create_tensor_2d(
                dw_ct_name(l), session->gguf_loader->get_tensor_type(src), cfg_.d_model,
                cfg_.conv_kernel_size);
        }
]=]
            [=[        for (int l = 0; l < cfg_.n_layers; l++) {
            const std::string src =
                name_ + ".layers." + std::to_string(l) + ".conv.depthwise_conv.weight";
            session->model_tensor_container->create_tensor_2d(
                dw_ct_name(l), session->gguf_loader->get_tensor_type(src), cfg_.d_model,
                cfg_.conv_kernel_size);
        }

        session->model_tensor_container->create_tensor_3d(
            "encoder.diag.pre", GGML_TYPE_F32, cfg_.d_model, cfg_.cache_chunk_frames, 1);
        session->model_tensor_container->create_tensor_3d(
            "encoder.diag.ff1_residual", GGML_TYPE_F32, cfg_.d_model,
            cfg_.cache_chunk_frames, 1);
        session->model_tensor_container->create_tensor_3d(
            "encoder.diag.attn_norm", GGML_TYPE_F32, cfg_.d_model,
            cfg_.cache_chunk_frames, 1);
        session->model_tensor_container->create_tensor_3d(
            "encoder.diag.qkv", GGML_TYPE_F32, 3 * cfg_.d_model, cfg_.cache_chunk_frames, 1);
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[        pre.set_first_tensor(ggml_runtime::ggml_bf_tensor(x_view, x.buft));
    }

    // Cache-aware layers consume the precomputed per-layer positional projection.
]=]
            [=[        pre.set_first_tensor(ggml_runtime::ggml_bf_tensor(x_view, x.buft));
    }

    if (pre.get_tensor(0).tensor->ne[2] == 1) {
        auto x = pre.get_tensor(0);
        auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(x.buft);
        auto dst = session->model_tensor_container->get_tensor_by_name("encoder.diag.pre");
        auto store = ggml_cpy(bf_ctx.ctx, x.tensor, dst.tensor);
        session_tensor_container->cache_tensor(
            "encoder.diag.pre.store", ggml_runtime::ggml_bf_tensor(store, dst.buft));
    }

    // Cache-aware layers consume the precomputed per-layer positional projection.
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[    residual = ggml_add(bf_ctx.ctx, residual, ggml_scale(bf_ctx.ctx, ff1.tensor, 0.5f));

    // MHA: non-cached uses RelPositionMultiHeadAttention; cached path builds
]=]
            [=[    residual = ggml_add(bf_ctx.ctx, residual, ggml_scale(bf_ctx.ctx, ff1.tensor, 0.5f));
    if (cache != nullptr && name_ == "encoder.layers.0" && residual->ne[2] == 1) {
        auto dst = session->model_tensor_container->get_tensor_by_name("encoder.diag.ff1_residual");
        auto store = ggml_cpy(bf_ctx.ctx, residual, dst.tensor);
        session_tensor_container->cache_tensor(
            "encoder.diag.ff1_residual.store",
            ggml_runtime::ggml_bf_tensor(store, dst.buft));
    }

    // MHA: non-cached uses RelPositionMultiHeadAttention; cached path builds
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[    auto x_normed = mha_in.get_tensor(0);

    ggml_tensor* attn_tensor = nullptr;
]=]
            [=[    auto x_normed = mha_in.get_tensor(0);
    if (cache != nullptr && name_ == "encoder.layers.0" && x_normed.tensor->ne[2] == 1) {
        auto dst = session->model_tensor_container->get_tensor_by_name("encoder.diag.attn_norm");
        auto store = ggml_cpy(bf_ctx.ctx, x_normed.tensor, dst.tensor);
        session_tensor_container->cache_tensor(
            "encoder.diag.attn_norm.store",
            ggml_runtime::ggml_bf_tensor(store, dst.buft));
    }

    ggml_tensor* attn_tensor = nullptr;
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[    ggml_tensor* qkv = add_bias(ggml_mul_mat(bf_ctx.ctx, wqkv.tensor, x.tensor), bqkv);
    const size_t qkv_es = ggml_element_size(qkv);
]=]
            [=[    ggml_tensor* qkv = add_bias(ggml_mul_mat(bf_ctx.ctx, wqkv.tensor, x.tensor), bqkv);
    if (name_ == "encoder.layers.0" && qkv->ne[2] == 1) {
        auto dst = session->model_tensor_container->get_tensor_by_name("encoder.diag.qkv");
        auto store = ggml_cpy(bf_ctx.ctx, qkv, dst.tensor);
        session_tensor_container->cache_tensor(
            "encoder.diag.qkv.store", ggml_runtime::ggml_bf_tensor(store, dst.buft));
    }
    const size_t qkv_es = ggml_element_size(qkv);
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[    ggml_runtime::TensorBag out;
    out.add_tensor(x_bag.get_tensor(0));
    // Device-resident cache feedback keeps state on the GPU between chunks.
]=]
            [=[    ggml_runtime::TensorBag out;
    out.add_tensor(x_bag.get_tensor(0));
    if (batch == 1) {
        for (const char* name : {
                 "encoder.diag.pre.store", "encoder.diag.ff1_residual.store",
                 "encoder.diag.attn_norm.store", "encoder.diag.qkv.store"}) {
            out.add_tensor(session_tensor_container->get_tensor_by_name(name));
        }
    }
    // Device-resident cache feedback keeps state on the GPU between chunks.
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[void
SubSampling::define_tensors(ggml_runtime::Session* session) {
    conv_->define_tensors(session);
    out_->define_tensors(session);
}
]=]
            [=[void
SubSampling::define_tensors(ggml_runtime::Session* session) {
    conv_->define_tensors(session);
    out_->define_tensors(session);

    const int C = cfg_.subsampling_conv_channels;
    int W = cfg_.feat_in;
    int H = 9 + cfg_.subsampling_factor * cfg_.cache_chunk_frames;
    for (int stage = 0; stage < 3; ++stage) {
        session->model_tensor_container->create_tensor_4d(
            "encoder.diag.sub.pad" + std::to_string(stage), GGML_TYPE_F32,
            W + 3, H + 3, stage == 0 ? 1 : C, 1);
        W = W / 2 + 1;
        H = H / 2 + 1;
        const int first_op = stage == 0 ? 0 : (stage == 1 ? 2 : 5);
        const int last_op = stage == 0 ? 1 : (stage == 1 ? 4 : 7);
        for (int op = first_op; op <= last_op; ++op) {
            session->model_tensor_container->create_tensor_4d(
                "encoder.diag.sub.op" + std::to_string(op), GGML_TYPE_F32, W, H, C, 1);
        }
    }
    session->model_tensor_container->create_tensor_2d(
        "encoder.diag.sub.flat", GGML_TYPE_F32, W * C, H);
    session->model_tensor_container->create_tensor_2d(
        "encoder.diag.sub.out", GGML_TYPE_F32, cfg_.d_model, H);
}
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[        ggml_runtime::TensorBag bag = input_tensors;
        for (auto* m : conv_->modules) {
]=]
            [=[        ggml_runtime::TensorBag bag = input_tensors;
        int module_index = 0;
        int pad_index = 0;
        for (auto* m : conv_->modules) {
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[                bag.set_first_tensor(ggml_runtime::ggml_bf_tensor(padded, t.buft));
            }
            bag = m->build_graph(session, bag, session_tensor_container);
        }
]=]
            [=[                bag.set_first_tensor(ggml_runtime::ggml_bf_tensor(padded, t.buft));
                auto dst = session->model_tensor_container->get_tensor_by_name(
                    "encoder.diag.sub.pad" + std::to_string(pad_index));
                auto store = ggml_cpy(bf_ctx.ctx, padded, dst.tensor);
                session_tensor_container->cache_tensor(
                    "encoder.diag.sub.pad" + std::to_string(pad_index) + ".store",
                    ggml_runtime::ggml_bf_tensor(store, dst.buft));
                ++pad_index;
            }
            bag = m->build_graph(session, bag, session_tensor_container);
            auto value = bag.get_tensor(0);
            auto value_ctx = session_tensor_container->get_ctx_of_buffer_type(value.buft);
            auto dst = session->model_tensor_container->get_tensor_by_name(
                "encoder.diag.sub.op" + std::to_string(module_index));
            auto store = ggml_cpy(value_ctx.ctx, value.tensor, dst.tensor);
            session_tensor_container->cache_tensor(
                "encoder.diag.sub.op" + std::to_string(module_index) + ".store",
                ggml_runtime::ggml_bf_tensor(store, dst.buft));
            ++module_index;
        }
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[    ggml_runtime::TensorBag output_tensors;
    output_tensors.add_tensor(ggml_runtime::ggml_bf_tensor(reshape_tensor, buft_ctx.buft));
    return out_->build_graph(session, output_tensors, session_tensor_container);
]=]
            [=[    ggml_runtime::TensorBag output_tensors;
    output_tensors.add_tensor(ggml_runtime::ggml_bf_tensor(reshape_tensor, buft_ctx.buft));
    auto flat_dst = session->model_tensor_container->get_tensor_by_name("encoder.diag.sub.flat");
    auto flat_store = ggml_cpy(buft_ctx.ctx, reshape_tensor, flat_dst.tensor);
    session_tensor_container->cache_tensor(
        "encoder.diag.sub.flat.store",
        ggml_runtime::ggml_bf_tensor(flat_store, flat_dst.buft));
    auto result = out_->build_graph(session, output_tensors, session_tensor_container);
    auto out_value = result.get_tensor(0);
    auto out_dst = session->model_tensor_container->get_tensor_by_name("encoder.diag.sub.out");
    auto out_store = ggml_cpy(buft_ctx.ctx, out_value.tensor, out_dst.tensor);
    session_tensor_container->cache_tensor(
        "encoder.diag.sub.out.store", ggml_runtime::ggml_bf_tensor(out_store, out_dst.buft));
    return result;
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[        for (const char* name : {
                 "encoder.diag.pre.store", "encoder.diag.ff1_residual.store",
                 "encoder.diag.attn_norm.store", "encoder.diag.qkv.store"}) {
            out.add_tensor(session_tensor_container->get_tensor_by_name(name));
        }
]=]
            [=[        for (const char* name : {
                 "encoder.diag.sub.pad0.store", "encoder.diag.sub.op0.store",
                 "encoder.diag.sub.op1.store", "encoder.diag.sub.pad1.store",
                 "encoder.diag.sub.op2.store", "encoder.diag.sub.op3.store",
                 "encoder.diag.sub.op4.store", "encoder.diag.sub.pad2.store",
                 "encoder.diag.sub.op5.store", "encoder.diag.sub.op6.store",
                 "encoder.diag.sub.op7.store", "encoder.diag.sub.flat.store",
                 "encoder.diag.sub.out.store", "encoder.diag.pre.store",
                 "encoder.diag.ff1_residual.store", "encoder.diag.attn_norm.store",
                 "encoder.diag.qkv.store"}) {
            out.add_tensor(session_tensor_container->get_tensor_by_name(name));
        }
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        file(WRITE "${_NEMO_FASTCONFORMER_CPP}" "${_NEMO_FASTCONFORMER_TEXT}")
    endif()
    if(NOT _NEMO_FASTCONFORMER_TEXT MATCHES "encoder.diag.final.store")
        string(REPLACE
            [=[        session->model_tensor_container->create_tensor_3d(
            "encoder.diag.qkv", GGML_TYPE_F32, 3 * cfg_.d_model, cfg_.cache_chunk_frames, 1);
]=]
            [=[        session->model_tensor_container->create_tensor_3d(
            "encoder.diag.qkv", GGML_TYPE_F32, 3 * cfg_.d_model, cfg_.cache_chunk_frames, 1);
        session->model_tensor_container->create_tensor_3d(
            "encoder.diag.final", GGML_TYPE_F32, cfg_.d_model, cfg_.cache_chunk_frames, 1);
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[    ggml_runtime::TensorBag out;
    out.add_tensor(x_bag.get_tensor(0));
]=]
            [=[    ggml_runtime::TensorBag out;
    auto final_value = x_bag.get_tensor(0);
    if (batch == 1) {
        auto final_ctx = session_tensor_container->get_ctx_of_buffer_type(final_value.buft);
        auto final_dst =
            session->model_tensor_container->get_tensor_by_name("encoder.diag.final");
        auto final_store = ggml_cpy(final_ctx.ctx, final_value.tensor, final_dst.tensor);
        session_tensor_container->cache_tensor(
            "encoder.diag.final.store",
            ggml_runtime::ggml_bf_tensor(final_store, final_dst.buft));
    }
    out.add_tensor(final_value);
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[                 "encoder.diag.sub.out.store", "encoder.diag.pre.store",
                 "encoder.diag.ff1_residual.store", "encoder.diag.attn_norm.store",
                 "encoder.diag.qkv.store"}) {
]=]
            [=[                 "encoder.diag.sub.out.store", "encoder.diag.pre.store",
                 "encoder.diag.ff1_residual.store", "encoder.diag.attn_norm.store",
                 "encoder.diag.qkv.store", "encoder.diag.final.store"}) {
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        file(WRITE "${_NEMO_FASTCONFORMER_CPP}" "${_NEMO_FASTCONFORMER_TEXT}")
    endif()
    if(NOT _NEMO_FASTCONFORMER_TEXT MATCHES "diag.output.store")
        string(REPLACE
            [=[    norm_out_->define_tensors(session);
}
]=]
            [=[    norm_out_->define_tensors(session);
    if (cfg_.cache_mode == CacheMode::Enabled) {
        session->model_tensor_container->create_tensor_3d(
            name_ + ".diag.output", GGML_TYPE_F32, cfg_.d_model,
            cfg_.cache_chunk_frames, 1);
    }
}
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[    ggml_runtime::TensorBag out;
    out.add_tensor(out_in.get_tensor(0));
    if (cache == nullptr) {
]=]
            [=[    ggml_runtime::TensorBag out;
    auto layer_output = out_in.get_tensor(0);
    if (cache != nullptr && layer_output.tensor->ne[2] == 1) {
        auto dst = session->model_tensor_container->get_tensor_by_name(name_ + ".diag.output");
        auto store = ggml_cpy(bf_ctx.ctx, layer_output.tensor, dst.tensor);
        session_tensor_container->cache_tensor(
            name_ + ".diag.output.store",
            ggml_runtime::ggml_bf_tensor(store, dst.buft));
    }
    out.add_tensor(layer_output);
    if (cache == nullptr) {
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        string(REPLACE
            [=[    // Device-resident cache feedback keeps state on the GPU between chunks.
]=]
            [=[    if (batch == 1) {
        for (int l = 0; l < cfg_.n_layers; ++l) {
            out.add_tensor(session_tensor_container->get_tensor_by_name(
                name_ + ".layers." + std::to_string(l) + ".diag.output.store"));
        }
    }
    // Device-resident cache feedback keeps state on the GPU between chunks.
]=]
            _NEMO_FASTCONFORMER_TEXT
            "${_NEMO_FASTCONFORMER_TEXT}"
        )
        file(WRITE "${_NEMO_FASTCONFORMER_CPP}" "${_NEMO_FASTCONFORMER_TEXT}")
    endif()
endif()

if(ANDROID AND VOXLINE_NEMOTRON_NUMERICAL_TRACE)
    # Temporary RNNT joint checkpoints. They isolate the small decoder tail
    # from the already-qualified encoder by retaining the first two-frame
    # activation and vocabulary logits in backend-owned tensors. Voxline reads
    # them after the first joint call; they do not participate in decoding.
    set(_NEMO_RNNT_MODULES_CPP
        "${nemo_speech_source_SOURCE_DIR}/src/asr/decoders/rnnt_modules.cpp")
    file(READ "${_NEMO_RNNT_MODULES_CPP}" _NEMO_RNNT_MODULES_TEXT)
    if(NOT _NEMO_RNNT_MODULES_TEXT MATCHES "rnnt.diag.logits.store")
        string(REPLACE
            [=[void
RnntJointModule::define_decoder_tensors(ggml_runtime::Session* session) {
    pred_proj_->define_tensors(session);
    out_proj_->define_tensors(session);
}
]=]
            [=[void
RnntJointModule::define_decoder_tensors(ggml_runtime::Session* session) {
    pred_proj_->define_tensors(session);
    out_proj_->define_tensors(session);
    session->model_tensor_container->create_tensor_2d(
        "rnnt.diag.act", GGML_TYPE_F32, cfg_.joint_dim, 2);
    session->model_tensor_container->create_tensor_2d(
        "rnnt.diag.logits", GGML_TYPE_F32, cfg_.joint_output_size(), 2);
}
]=]
            _NEMO_RNNT_MODULES_TEXT
            "${_NEMO_RNNT_MODULES_TEXT}"
        )
        string(REPLACE
            [=[    if (argmax_only) {
        auto l = logits.get_tensor(0);
]=]
            [=[    if (argmax_only) {
        auto l = logits.get_tensor(0);
        const bool capture_diag = B == 1 && l.tensor->ne[1] == 2;
        if (capture_diag) {
            auto act_dst = session->model_tensor_container->get_tensor_by_name("rnnt.diag.act");
            auto logits_dst =
                session->model_tensor_container->get_tensor_by_name("rnnt.diag.logits");
            auto act_store = ggml_cpy(bf_ctx.ctx, act, act_dst.tensor);
            auto logits_store = ggml_cpy(bf_ctx.ctx, l.tensor, logits_dst.tensor);
            tc->cache_tensor(
                "rnnt.diag.act.store",
                ggml_runtime::ggml_bf_tensor(act_store, act_dst.buft));
            tc->cache_tensor(
                "rnnt.diag.logits.store",
                ggml_runtime::ggml_bf_tensor(logits_store, logits_dst.buft));
        }
]=]
            _NEMO_RNNT_MODULES_TEXT
            "${_NEMO_RNNT_MODULES_TEXT}"
        )
        string(REPLACE
            [=[            out.add_tensor(ggml_runtime::ggml_bf_tensor(flat_ids, bf_ctx.buft));
            return out;
]=]
            [=[            out.add_tensor(ggml_runtime::ggml_bf_tensor(flat_ids, bf_ctx.buft));
            if (capture_diag) {
                out.add_tensor(tc->get_tensor_by_name("rnnt.diag.act.store"));
                out.add_tensor(tc->get_tensor_by_name("rnnt.diag.logits.store"));
            }
            return out;
]=]
            _NEMO_RNNT_MODULES_TEXT
            "${_NEMO_RNNT_MODULES_TEXT}"
        )
        file(WRITE "${_NEMO_RNNT_MODULES_CPP}" "${_NEMO_RNNT_MODULES_TEXT}")
    endif()
endif()

if(ANDROID)
    # Keep the tiny scalar log transform at the host output boundary. Several
    # accelerator backends correctly execute the expensive DFT/mel projection
    # but do not implement LOG, which otherwise creates a cross-backend graph.
    # Reusing that split graph after scheduler reset can retain stale device
    # placement metadata. A host log is backend-neutral and only touches the
    # already-returned n_mels * n_frames values.
    set(_NEMO_FE_CPP "${nemo_speech_source_SOURCE_DIR}/src/asr/features/fe.cpp")
    file(READ "${_NEMO_FE_CPP}" _NEMO_FE_TEXT)
    if(NOT _NEMO_FE_TEXT MATCHES "fe.log_mel_on_host")
        string(REPLACE
            [=[        ggml_tensor* mel_eps = ggml_scale_bias(g, mel, 1.0f, cfg_.log_zero_guard);
        ggml_tensor* log_mel = ggml_log(g, mel_eps);
        ggml_set_name(log_mel, "log_mel");

        ggml_runtime::TensorBag out;
        out.add_tensor(ggml_runtime::ggml_bf_tensor(log_mel, x.buft));
]=]
            [=[        ggml_tensor* mel_eps = ggml_scale_bias(g, mel, 1.0f, cfg_.log_zero_guard);
        ggml_set_name(mel_eps, "fe.log_mel_on_host");

        ggml_runtime::TensorBag out;
        out.add_tensor(ggml_runtime::ggml_bf_tensor(mel_eps, x.buft));
]=]
            _NEMO_FE_TEXT
            "${_NEMO_FE_TEXT}"
        )
        string(REPLACE
            [=[        features.assign(packed.begin() + b * output_item, packed.begin() + (b + 1) * output_item);
        const int valid = std::min(static_cast<int>(n_samples) / hop, n_frames);
]=]
            [=[        features.assign(packed.begin() + b * output_item, packed.begin() + (b + 1) * output_item);
        for (float& value : features) value = std::log(value);
        const int valid = std::min(static_cast<int>(n_samples) / hop, n_frames);
]=]
            _NEMO_FE_TEXT
            "${_NEMO_FE_TEXT}"
        )
        file(WRITE "${_NEMO_FE_CPP}" "${_NEMO_FE_TEXT}")
    endif()
endif()

set(NEMO_SPEECH_DIR "${nemo_speech_source_SOURCE_DIR}")
file(REMOVE_RECURSE "${NEMO_SPEECH_DIR}/ggml")
file(CREATE_LINK "${ggml_source_SOURCE_DIR}" "${NEMO_SPEECH_DIR}/ggml" SYMBOLIC)

if(ANDROID AND GGML_CPU_ALL_VARIANTS)
    # NeMo's install rule assumes the single static CPU target exists. ggml's
    # dispatch build replaces it with ggml-cpu-android_* modules, so keep the
    # irrelevant install rule from referring to a non-existent target.
    set(_NEMO_ROOT_CMAKE "${NEMO_SPEECH_DIR}/CMakeLists.txt")
    file(READ "${_NEMO_ROOT_CMAKE}" _NEMO_ROOT_CMAKE_TEXT)
    string(REPLACE
        "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY \${CMAKE_BINARY_DIR}/bin)\nset(CMAKE_LIBRARY_OUTPUT_DIRECTORY \${CMAKE_BINARY_DIR}/bin)"
        "if(NOT ANDROID)\n    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY \${CMAKE_BINARY_DIR}/bin)\n    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY \${CMAKE_BINARY_DIR}/bin)\nendif()"
        _NEMO_ROOT_CMAKE_TEXT
        "${_NEMO_ROOT_CMAKE_TEXT}"
    )
    string(REPLACE
        "set(GGML_DEPENDENCIES ggml ggml-base ggml-cpu)"
        "set(GGML_DEPENDENCIES ggml ggml-base)"
        _NEMO_ROOT_CMAKE_TEXT
        "${_NEMO_ROOT_CMAKE_TEXT}"
    )
    string(REPLACE
        "install(TARGETS ggml ggml-base ggml-cpu\n"
        "install(TARGETS ggml ggml-base\n"
        _NEMO_ROOT_CMAKE_TEXT
        "${_NEMO_ROOT_CMAKE_TEXT}"
    )
    string(REPLACE
        "if(GGML_VULKAN)\n    list(APPEND GGML_DEPENDENCIES ggml-vulkan)\nendif()"
        "if(GGML_VULKAN AND NOT ANDROID)\n    list(APPEND GGML_DEPENDENCIES ggml-vulkan)\nendif()"
        _NEMO_ROOT_CMAKE_TEXT
        "${_NEMO_ROOT_CMAKE_TEXT}"
    )
    file(WRITE "${_NEMO_ROOT_CMAKE}" "${_NEMO_ROOT_CMAKE_TEXT}")
endif()

# NeMo-Speech.cpp currently assumes it is the CMake root in a few include paths.
set(VOXLINE_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(CMAKE_SOURCE_DIR "${NEMO_SPEECH_DIR}")
add_subdirectory("${NEMO_SPEECH_DIR}" nemo-speech EXCLUDE_FROM_ALL)
set(CMAKE_SOURCE_DIR "${VOXLINE_CMAKE_SOURCE_DIR}")
target_include_directories(nemo_speech_asr_c PUBLIC "${NEMO_SPEECH_DIR}/include")
if(ANDROID)
    # sentencepiece's protobuf-lite logger calls __android_log_write.
    target_link_libraries(nemo_speech_asr PRIVATE log)
endif()

if(ANDROID)
    add_library(
        voxline_nemotron
        SHARED
        nemotron_jni.cpp
        nemotron_runtime.cpp
    )
    target_include_directories(voxline_nemotron PRIVATE "${NEMO_SPEECH_DIR}/include")
    target_link_libraries(
        voxline_nemotron
        PRIVATE
        nemo_speech_asr
        ggml
        ggml-base
        log
    )
    if(TARGET ggml-opencl)
        add_dependencies(voxline_nemotron ggml-opencl)
    endif()
elseif(VOXLINE_BUILD_HOST_NEMOTRON_TRACE)
    function(voxline_configure_nemotron_host_target target)
        target_include_directories(
            ${target}
            PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${NEMO_SPEECH_DIR}/include"
        )
        target_link_libraries(
            ${target}
            PRIVATE
            nemo_speech_asr
            ggml
            ggml-base
        )
    endfunction()

    add_executable(
        nemotron_host_trace
        nemotron_runtime.cpp
        "${CMAKE_CURRENT_SOURCE_DIR}/../../test/cpp/nemotron_host_trace.cpp"
    )
    voxline_configure_nemotron_host_target(nemotron_host_trace)

    add_executable(
        nemotron_runtime_compare
        nemotron_runtime.cpp
        "${CMAKE_CURRENT_SOURCE_DIR}/../../test/cpp/nemotron_runtime_compare.cpp"
    )
    voxline_configure_nemotron_host_target(nemotron_runtime_compare)

    add_executable(
        nemotron_streaming_policy_test
        "${CMAKE_CURRENT_SOURCE_DIR}/../../test/cpp/nemotron_streaming_policy_test.cpp"
    )
    target_include_directories(
        nemotron_streaming_policy_test
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    enable_testing()
    add_test(NAME nemotron_streaming_policy COMMAND nemotron_streaming_policy_test)
endif()
