#pragma once
#include <maya/MOpenCLInfo.h>
#include <maya/MPxGPUDeformer.h>


// Kernel
class OpenCLKernel
{
private:
	MAutoCLKernel m_kernel;
	size_t m_localWorkSize = 0;
	size_t m_globalWorkSize = 0;

public:
	OpenCLKernel() = default;
	~OpenCLKernel() = default;

	MStatus Initialize(const MString& kernelFilePath, const MString& kernelName, uint32_t numVertices);
	void Finalize();

	cl_kernel Get() const;
	MStatus Run(const std::vector<cl_event>& inEvents, cl_event* outEvent);
};