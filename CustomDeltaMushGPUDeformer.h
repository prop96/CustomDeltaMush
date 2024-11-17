#pragma once

#include "OpenCLKernel.h"
#include "DeltaMushBindMeshData.h"
#include <maya/MPxGPUDeformer.h>
#include <maya/MGPUDeformerRegistry.h>
#include <maya/MDataBlock.h>
#include <maya/MOpenCLInfo.h>
#include <maya/MPointArray.h>
#include <clew/clew_cl.h>
#include <vector>


// The GPU override implementation of the CustomDeltaMushDeformer node
class CustomDeltaMushGPUDeformer : public MPxGPUDeformer
{
public:
	static MGPUDeformerRegistrationInfo* getGPUDeformerInfo();
	static bool validateNodeInGraph(MDataBlock& block, const MEvaluationNode&, const MPlug& plug, MStringArray* messages);
	static bool validateNodeValues(MDataBlock& block, const MEvaluationNode&, const MPlug& plug, MStringArray* messages);

	CustomDeltaMushGPUDeformer() = default;
	~CustomDeltaMushGPUDeformer() override;

	// Implementation of MPxGPUDeformer.
	void terminate() override;
	MPxGPUDeformer::DeformerStatus evaluate(
		MDataBlock& block,
		const MEvaluationNode& evaluationNode,
		const MPlug& outputPlug,
		const MPlugArray& inputPlugs,
		const MGPUDeformerData& inputData,
		MGPUDeformerData& outputData) override;

private:
	DeltaMushBindMeshData m_bindMeshData;

	OpenCLKernel m_smoothingKernel;
	OpenCLKernel m_applyDeltaKernel;
	int32_t m_size = 0;

	// neighboring vertex indices for Laplacian smoothing
	std::vector<int32_t> m_neighbourIndices;
	MAutoCLMem m_neighbourIndicesBuffer;

	// temporary buffers for iterative smoothing
	cl_mem m_tmpBuffer0 = nullptr;
	cl_mem m_tmpBuffer1 = nullptr;

	// delta computed for the bound mesh
	std::vector<float> m_originalDelta;
	MAutoCLMem m_originalDeltaBuffer;

	// delta length computed for the bound mesh
	std::vector<float> m_deltaLength;
	MAutoCLMem m_deltaLengthBuffer;
};

// registration information for the GPU deformer
class CustomDeltaMushGPUDeformerInfo : public MGPUDeformerRegistrationInfo
{
public:
	CustomDeltaMushGPUDeformerInfo() {}
	virtual ~CustomDeltaMushGPUDeformerInfo() {}

	MPxGPUDeformer* createGPUDeformer() override
	{
		return new CustomDeltaMushGPUDeformer();
	}

	bool validateNodeInGraph(
		MDataBlock& block,
		const MEvaluationNode& evaluationNode,
		const MPlug& plug,
		MStringArray* messages) override
	{
		return CustomDeltaMushGPUDeformer::validateNodeInGraph(block, evaluationNode, plug, messages);
	}

	bool validateNodeValues(
		MDataBlock& block,
		const MEvaluationNode& evaluationNode,
		const MPlug& plug,
		MStringArray* messages) override
	{
		return CustomDeltaMushGPUDeformer::validateNodeValues(block, evaluationNode, plug, messages);
	}
};
