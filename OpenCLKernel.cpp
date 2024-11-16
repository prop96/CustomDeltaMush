#include "OpenCLKernel.h"
#include <maya/MGlobal.h>

MStatus OpenCLKernel::Initialize(const MString& kernelFilePath, const MString& kernelName, uint32_t numVertices)
{
	cl_int err = CL_SUCCESS;

	m_kernel = MOpenCLInfo::getOpenCLKernel(kernelFilePath, kernelName);

	if (m_kernel.isNull())
	{
		MGlobal::displayError("error getting average kernel from file");
		return MStatus::kFailure;
	}

	// Figure out a good work group size for our kernel
	m_localWorkSize = 0;
	m_globalWorkSize = 0;
	size_t retSize = 0;
	err = clGetKernelWorkGroupInfo(
		m_kernel.get(),
		MOpenCLInfo::getOpenCLDeviceId(),
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(size_t),
		&m_localWorkSize,
		&retSize
	);
	MOpenCLInfo::checkCLErrorStatus(err);
	if (err != CL_SUCCESS || retSize == 0 || m_localWorkSize == 0)
	{
		return MStatus::kFailure;
	}

	// Global work size must be a multiple of local work size
	const size_t remain = numVertices % m_localWorkSize;
	if (remain != 0)
	{
		m_globalWorkSize = numVertices + (m_localWorkSize - remain);
	}
	else
	{
		m_globalWorkSize = numVertices;
	}

	return MStatus::kSuccess;
}

void OpenCLKernel::Finalize()
{
	MOpenCLInfo::releaseOpenCLKernel(m_kernel);
	m_kernel.reset();
}

cl_kernel OpenCLKernel::Get() const
{
	return m_kernel.get();
}

MStatus OpenCLKernel::Run(const std::vector<cl_event>& inEvents, cl_event* outEvent)
{
	const uint32_t inEventCount = inEvents.size();

	cl_int err = clEnqueueNDRangeKernel(
		MOpenCLInfo::getMayaDefaultOpenCLCommandQueue(),
		m_kernel.get(),
		1,
		nullptr,
		&m_globalWorkSize,
		&m_localWorkSize,
		inEventCount,
		inEventCount == 0 ? nullptr : inEvents.data(),
		outEvent
	);

	if (err != CL_SUCCESS)
	{
		MGlobal::displayError("Error: failed to call clEnqueueNDRangeKernel");
		return MStatus::kFailure;
	}

	return MStatus::kSuccess;
}
