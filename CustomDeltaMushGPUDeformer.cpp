#include "CustomDeltaMushGPUDeformer.h"
#include "CustomDeltaMushDeformer.h"
#include <maya/MTypeId.h> 
#include <maya/MStringArray.h>
#include <maya/MGlobal.h>
#include <maya/MItGeometry.h>
#include <maya/MOpenCLInfo.h>
#include <clew/clew.h>
#include <cassert>

constexpr int MAX_NEIGH = 4;


MGPUDeformerRegistrationInfo* CustomDeltaMushGPUDeformer::getGPUDeformerInfo()
{
	static CustomDeltaMushGPUDeformerInfo theOne;
	return &theOne;
}

CustomDeltaMushGPUDeformer::~CustomDeltaMushGPUDeformer()
{
	terminate();
}

bool CustomDeltaMushGPUDeformer::validateNodeInGraph(MDataBlock& block, const MEvaluationNode& evaluationNode, const MPlug& plug, MStringArray* messages)
{
	// Support everything.
	return true;
}

bool CustomDeltaMushGPUDeformer::validateNodeValues(MDataBlock& block, const MEvaluationNode& evaluationNode, const MPlug& plug, MStringArray* messages)
{
	// Support everything.
	return true;
}

void CustomDeltaMushGPUDeformer::terminate()
{
	m_neighbourIndicesBuffer.reset();
	m_originalDeltaBuffer.reset();
	m_deltaLengthBuffer.reset();

	delete(m_tmpBuffer0);
	delete(m_tmpBuffer1);

	m_smoothingKernel.Finalize();
	m_applyDeltaKernel.Finalize();
}

MPxGPUDeformer::DeformerStatus CustomDeltaMushGPUDeformer::evaluate(
	MDataBlock& block,
	const MEvaluationNode& evaluationNode,
	const MPlug& plug,
	const MPlugArray& inputPlugs,
	const MGPUDeformerData& inputData,
	MGPUDeformerData& outputData)
{
	// get inputPositions from input mesh
	{
		const uint32_t numPlugs = inputPlugs.length();
		assert(numPlugs == 0);
	}
	const MPlug& inputPlug = inputPlugs[0];
	const MGPUDeformerBuffer inputPositions = inputData.getBuffer(MPxGPUDeformer::sPositionsName(), inputPlug);
	if (!inputPositions.isValid())
	{
		return kDeformerFailure;
	}

	// create outputPositions buffer
	MGPUDeformerBuffer outputPositions = createOutputBuffer(inputPositions);
	if (outputPositions.isValid())
	{
		return kDeformerFailure;
	}

	// Getting needed data
	double applyDeltaV = block.inputValue(CustomDeltaMushDeformer::applyDelta).asDouble();
	double amountV = block.inputValue(CustomDeltaMushDeformer::smoothAmount).asDouble();
	bool rebintV = block.inputValue(CustomDeltaMushDeformer::rebind).asBool();

	const int32_t iterationsVal = block.inputValue(CustomDeltaMushDeformer::smoothIterations).asInt();

	// nothing to do if no smoothing
	if (iterationsVal == 0)
	{
		return kDeformerSuccess;
	}


	// # of vertices?
	const uint32_t numElements = inputPositions.elementCount();
	cl_int err = CL_SUCCESS;

	DeformerStatus dstatus;

	// set up openCL kernel
	const MString kernelFile(CustomDeltaMushDeformer::pluginPath + "/deltaMush.cl");
	if (!m_smoothingKernel.Get())
	{
		MString kernelName("CreateMush");
		m_smoothingKernel.Initialize(kernelFile, kernelName, numElements);
	}
	if (!m_applyDeltaKernel.Get())
	{
		MString kernelName("ApplyDelta");
		m_applyDeltaKernel.Initialize(kernelFile, kernelName, numElements);
	}

	{
		// init data builds the neighbour table and we are going to upload it
		MObject referenceMeshV = block.inputValue(CustomDeltaMushDeformer::outputGeom).data();

		int size = numElements;
		m_size = size;
		m_neighbourIndices.resize(size * MAX_NEIGH);
		m_originalDelta.resize(size * 3 * (MAX_NEIGH - 1));
		m_deltaLength.resize(size);

		// TODO: set up vectors here
		//RebindData(referenceMeshV, iterationsVal, amountV);

		// create buffers and upload data
		cl_int clStatus;

		m_neighbourIndicesBuffer.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
			m_neighbourIndices.size() * sizeof(uint32_t), m_neighbourIndices.data(), &clStatus));
		MOpenCLInfo::checkCLErrorStatus(clStatus);

		m_tmpBuffer0 = clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_READ_ONLY,
			3 * size * sizeof(float), nullptr, &clStatus);
		MOpenCLInfo::checkCLErrorStatus(clStatus);

		m_tmpBuffer1 = clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_READ_ONLY,
			3 * size * sizeof(float), nullptr, &clStatus);
		MOpenCLInfo::checkCLErrorStatus(clStatus);

		m_originalDeltaBuffer.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
			m_originalDelta.size() * sizeof(float), m_originalDelta.data(), &clStatus));
		MOpenCLInfo::checkCLErrorStatus(clStatus);

		m_deltaLengthBuffer.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
			m_deltaLength.size() * sizeof(float), m_deltaLength.data(), &clStatus));
		MOpenCLInfo::checkCLErrorStatus(clStatus);
	}

	// Set up our input events.  The input event could be NULL, in that case we need to pass
	// slightly different parameters into clEnqueueNDRangeKernel.
	std::vector<cl_event> events = { 0 };
	cl_uint eventCount = 0;
	if (inputPositions.bufferReadyEvent().get())
	{
		events[eventCount++] = inputPositions.bufferReadyEvent().get();
	}

	void* src = (void*)&m_tmpBuffer0;
	void* trg = (void*)inputPositions.buffer().getReadOnlyRef();

	cl_event inEvent;
	cl_event outEvent;
	for (int i = 0; i < iterationsVal; i++)
	{
		// Swap src and trg
		{
			void* tmpPtr = src;
			src = trg;
			trg = tmpPtr;
		}

		if (i == 1)
		{
			trg = (void*)&m_tmpBuffer1;
		}

		// Set all of our kernel parameters.
		uint32_t parameterId = 0;
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_mem), trg);
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_mem), m_neighbourIndicesBuffer.getReadOnlyRef());
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_mem), src);
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_float), (void*)&amountV);
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_uint), (void*)&i);
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_uint), (void*)&numElements);
		MOpenCLInfo::checkCLErrorStatus(err);

		// Run the Kernel
		if (i == 0)
		{
			err = m_smoothingKernel.Run(events, &outEvent);
			inEvent = events[0];
		}
		else
		{
			err = m_smoothingKernel.Run({ inEvent }, &outEvent);
		}
		MOpenCLInfo::checkCLErrorStatus(err);

		cl_event tmp = outEvent;
		outEvent = inEvent;
		inEvent = tmp;
	}

	uint32_t parameterId = 0;
	err = clSetKernelArg(m_applyDeltaKernel.Get(), parameterId++, sizeof(cl_mem), outputPositions.buffer().getReadOnlyRef());
	MOpenCLInfo::checkCLErrorStatus(err);
	err = clSetKernelArg(m_applyDeltaKernel.Get(), parameterId++, sizeof(cl_mem), m_originalDeltaBuffer.getReadOnlyRef());
	MOpenCLInfo::checkCLErrorStatus(err);
	err = clSetKernelArg(m_applyDeltaKernel.Get(), parameterId++, sizeof(cl_mem), m_deltaLengthBuffer.getReadOnlyRef());
	MOpenCLInfo::checkCLErrorStatus(err);
	err = clSetKernelArg(m_applyDeltaKernel.Get(), parameterId++, sizeof(cl_mem), m_neighbourIndicesBuffer.getReadOnlyRef());
	MOpenCLInfo::checkCLErrorStatus(err);
	err = clSetKernelArg(m_applyDeltaKernel.Get(), parameterId++, sizeof(cl_mem), trg);
	MOpenCLInfo::checkCLErrorStatus(err);
	err = clSetKernelArg(m_applyDeltaKernel.Get(), parameterId++, sizeof(cl_uint), (void*)&numElements);
	MOpenCLInfo::checkCLErrorStatus(err);

	// Run the Kernel
	MAutoCLEvent m_applyDeltaKernelFinishedEvent;
	err = m_applyDeltaKernel.Run( {}, m_applyDeltaKernelFinishedEvent.getReferenceForAssignment());

	outputPositions.setBufferReadyEvent(m_applyDeltaKernelFinishedEvent);
	MOpenCLInfo::checkCLErrorStatus(err);

	outputData.setBuffer(outputPositions);

	return kDeformerSuccess;
}
