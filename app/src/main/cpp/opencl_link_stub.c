// Link-only OpenCL ABI stub for Android builds.
//
// The real implementation is supplied by the device's public libOpenCL.so.
// This target exists only so the NDK linker can record DT_NEEDED=libOpenCL.so
// for libggml-opencl.so; Gradle explicitly excludes this stub from the APK.
// The functions are intentionally never executed.

#define OPENCL_STUB(name) void *name(void) { return (void *)0; }

OPENCL_STUB(clBuildProgram)
OPENCL_STUB(clCreateBuffer)
OPENCL_STUB(clCreateBufferWithProperties)
OPENCL_STUB(clCreateCommandQueue)
OPENCL_STUB(clCreateContext)
OPENCL_STUB(clCreateImage)
OPENCL_STUB(clCreateKernel)
OPENCL_STUB(clCreateProgramWithSource)
OPENCL_STUB(clCreateSubBuffer)
OPENCL_STUB(clEnqueueBarrierWithWaitList)
OPENCL_STUB(clEnqueueCopyBuffer)
OPENCL_STUB(clEnqueueFillBuffer)
OPENCL_STUB(clEnqueueMarkerWithWaitList)
OPENCL_STUB(clEnqueueNDRangeKernel)
OPENCL_STUB(clEnqueueReadBuffer)
OPENCL_STUB(clEnqueueWriteBuffer)
OPENCL_STUB(clFinish)
OPENCL_STUB(clFlush)
OPENCL_STUB(clGetDeviceIDs)
OPENCL_STUB(clGetDeviceInfo)
OPENCL_STUB(clGetEventProfilingInfo)
OPENCL_STUB(clGetKernelInfo)
OPENCL_STUB(clGetKernelSubGroupInfo)
OPENCL_STUB(clGetKernelWorkGroupInfo)
OPENCL_STUB(clGetPlatformIDs)
OPENCL_STUB(clGetPlatformInfo)
OPENCL_STUB(clGetProgramBuildInfo)
OPENCL_STUB(clReleaseContext)
OPENCL_STUB(clReleaseEvent)
OPENCL_STUB(clReleaseMemObject)
OPENCL_STUB(clReleaseProgram)
OPENCL_STUB(clSetKernelArg)
OPENCL_STUB(clWaitForEvents)
