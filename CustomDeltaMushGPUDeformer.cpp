#include "CustomDeltaMushGPUDeformer.h"
#include "CustomDeltaMushDeformer.h"
#include <maya/MTypeId.h> 
#include <maya/MStringArray.h>
#include <maya/MGlobal.h>
#include <maya/MItGeometry.h>
#include <maya/MOpenCLInfo.h>
#include <clew/clew.h>
#include <cassert>



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
	m_startNeighbourIndicesBuffer.reset();
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
	const MPlug& outputPlug,
	const MPlugArray& inputPlugs,
	const MGPUDeformerData& inputData,
	MGPUDeformerData& outputData)
{
	MStatus returnStat;

	// get inputPositions from input mesh
	{
		const uint32_t numPlugs = inputPlugs.length();
		assert(numPlugs == 1);
	}
	const MPlug& inputPlug = inputPlugs[0];
	const MGPUDeformerBuffer inputPositions = inputData.getBuffer(MPxGPUDeformer::sPositionsName(), inputPlug);
	if (!inputPositions.isValid())
	{
		return kDeformerFailure;
	}

	// create outputPositions buffer
	MGPUDeformerBuffer outputPositions = createOutputBuffer(inputPositions);
	if (!outputPositions.isValid())
	{
		return kDeformerFailure;
	}

	// set smoothing property
	{
		const int32_t iterationsVal = block.inputValue(CustomDeltaMushDeformer::smoothIterations).asInt();
		const double amountVal = block.inputValue(CustomDeltaMushDeformer::smoothAmount).asDouble();
		m_bindMeshData.SetSmoothingData(iterationsVal, amountVal);

		// nothing to do if no smoothing
		if (iterationsVal <= 0)
		{
			return kDeformerSuccess;
		}
	}

	// error if originalGeometry attribute is not connected
	MObject thisMObject = evaluationNode.dependencyNode(&returnStat);
	MFnDependencyNode thisNode(thisMObject);
	MObject origGeom = thisNode.attribute("originalGeometry", &returnStat);
	if (MPlug refMeshPlug(thisMObject, origGeom); !refMeshPlug.elementByLogicalIndex(0).isConnected(&returnStat))
	{
		returnStat.perror("mesh to bind is not connected");
		return MPxGPUDeformer::DeformerStatus::kDeformerFailure;
	}

	// initialize data for the mesh to bind if still not
	if (!m_bindMeshData.IsInitialized() || evaluationNode.dirtyPlugExists(origGeom))
	{
		// bind the original mesh
		MObject origGeomVal = block.inputArrayValue(origGeom, &returnStat).inputValue().asMesh();
		m_bindMeshData.SetBindMeshData(origGeomVal);

		block.inputValue(CustomDeltaMushDeformer::rebind).setBool(false);
	}

	// Getting needed data
	const double applyDeltaVal = block.inputValue(CustomDeltaMushDeformer::applyDelta).asDouble();

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

	// create buffers and upload data
	{
		cl_int clStatus;

		m_startNeighbourIndicesBuffer.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
			m_bindMeshData.GetStartIndexNeighbourIndices().size() * sizeof(uint32_t), (void*)m_bindMeshData.GetStartIndexNeighbourIndices().data(), &clStatus));
		MOpenCLInfo::checkCLErrorStatus(clStatus);

		m_neighbourIndicesBuffer.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
			m_bindMeshData.GetNeighbourIndices().size() * sizeof(int32_t), (void*)m_bindMeshData.GetNeighbourIndices().data(), &clStatus));
		MOpenCLInfo::checkCLErrorStatus(clStatus);

		m_tmpBuffer0 = clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_READ_ONLY,
			3 * numElements * sizeof(float), nullptr, &clStatus);
		MOpenCLInfo::checkCLErrorStatus(clStatus);

		m_tmpBuffer1 = clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_READ_ONLY,
			3 * numElements * sizeof(float), nullptr, &clStatus);
		MOpenCLInfo::checkCLErrorStatus(clStatus);

		m_originalDeltaBuffer.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
			m_bindMeshData.GetDelta().size() * sizeof(std::array<float, 3>), (void*)m_bindMeshData.GetDelta().data(), &clStatus));
		MOpenCLInfo::checkCLErrorStatus(clStatus);

		m_deltaLengthBuffer.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
			m_bindMeshData.GetDeltaLength().size() * sizeof(float), (void*)m_bindMeshData.GetDeltaLength().data(), &clStatus));
		MOpenCLInfo::checkCLErrorStatus(clStatus);
	}

	// smoothing process
	void* src = (void*)&m_tmpBuffer0;
	void* trg = (void*)inputPositions.buffer().getReadOnlyRef();
	MAutoCLEvent inEvent, outEvent;
	auto& smoothingData = m_bindMeshData.GetSmoothingData();
	for (uint32_t smoothItr = 0; smoothItr <  smoothingData.Iter; smoothItr++)
	{
		// Swap src and trg buffer
		{
			if (smoothItr == 1)
			{
				src = (void*)&m_tmpBuffer1;
			}

			void* tmpPtr = src;
			src = trg;
			trg = tmpPtr;
		}

		// swap the input and output event
		inEvent.swap(outEvent);

		// Set all of our kernel parameters.
		uint32_t parameterId = 0;
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_mem), trg);
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_mem), m_startNeighbourIndicesBuffer.getReadOnlyRef());
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_mem), m_neighbourIndicesBuffer.getReadOnlyRef());
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_mem), src);
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_float), (void*)&smoothingData.Amount);
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_uint), (void*)&numElements);
		MOpenCLInfo::checkCLErrorStatus(err);

		// Run the Kernel
		if (smoothItr == 0)
		{
			err = m_smoothingKernel.Run(inputPositions.bufferReadyEvent(), outEvent);
		}
		else
		{
			err = m_smoothingKernel.Run(inEvent, outEvent);
		}
		MOpenCLInfo::checkCLErrorStatus(err);
	}

	// adding delta process
	{
		uint32_t parameterId = 0;
		err = clSetKernelArg(m_applyDeltaKernel.Get(), parameterId++, sizeof(cl_mem), outputPositions.buffer().getReadOnlyRef());
		MOpenCLInfo::checkCLErrorStatus(err);
		err = clSetKernelArg(m_smoothingKernel.Get(), parameterId++, sizeof(cl_mem), m_startNeighbourIndicesBuffer.getReadOnlyRef());
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
		err = m_applyDeltaKernel.Run(outEvent, m_applyDeltaKernelFinishedEvent);

		outputPositions.setBufferReadyEvent(m_applyDeltaKernelFinishedEvent);
		MOpenCLInfo::checkCLErrorStatus(err);
	}

	outputData.setBuffer(outputPositions);

	return kDeformerSuccess;
}
