//  File: identityNode.cpp
//
//  Description:
//      Empty implementation of a deformer. This node
//      performs no deformation and is basically an empty
//      shell that can be used to create actual deformers.
//
//
//      Use this script to create a simple example with the identity node.
//      
//      loadPlugin identityNode;
//      
//      polyTorus -r 1 -sr 0.5 -tw 0 -sx 50 -sy 50 -ax 0 1 0 -cuv 1 -ch 1;
//      deformer -type "identity";
//      setKeyframe -v 0 -at weightList[0].weights[0] -t 1 identity1;
//      setKeyframe -v 1 -at weightList[0].weights[0] -t 60 identity1;
//      select -cl;

#include "CustomDeltaMushDeformerOpenCL.h"
#include "CustomDeltaMushDeformer.h"
#include <maya/MTypeId.h> 

#include <maya/MStringArray.h>
#include <maya/MGlobal.h>
#include <maya/MItGeometry.h>
#include <maya/MOpenCLInfo.h>
#include <clew/clew.h>

constexpr int MAX_NEIGH = 4;


MGPUDeformerRegistrationInfo* CustomDeltaMushGPUDeformer::getGPUDeformerInfo()
{
    static CustomDeltaMushGPUDeformerInfo theOne;
    return &theOne;
}

CustomDeltaMushGPUDeformer::CustomDeltaMushGPUDeformer() = default;

CustomDeltaMushGPUDeformer::~CustomDeltaMushGPUDeformer()
{
    terminate();
}

void* CustomDeltaMushGPUDeformer::creator()
{
    return new CustomDeltaMushGPUDeformer();
}

MStatus CustomDeltaMushGPUDeformer::initialize()
{
    return MStatus();
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

MPxGPUDeformer::DeformerStatus CustomDeltaMushGPUDeformer::evaluate(
    MDataBlock& block,
    const MEvaluationNode& evaluationNode,
    const MPlug& plug,
    const MPlugArray& inputPlugs,
    const MGPUDeformerData& inputData,
    MGPUDeformerData& outputData
)
{
    // Getting needed data
    float applyDeltaV = block.inputValue(CustomDeltaMushDeformer::applyDelta).asDouble();
    float amountV = block.inputValue(CustomDeltaMushDeformer::smoothAmount).asDouble();
    bool rebintV = block.inputValue(CustomDeltaMushDeformer::rebind).asBool();
    int iterationsV = block.inputValue(CustomDeltaMushDeformer::smoothIterations).asInt();

    int numPlugs = inputPlugs.length();
    const MPlug& inputPlug = inputPlugs[0];

    const MGPUDeformerBuffer inputPositions = inputData.getBuffer(MPxGPUDeformer::sPositionsName(), inputPlug);
    MGPUDeformerBuffer outputPositions = createOutputBuffer(inputPositions);
    if (!inputPositions.isValid() || outputPositions.isValid())
    {
        return kDeformerFailure;
    }

    // # of vertices?
    uint32_t numElements = inputPositions.elementCount();
    cl_int err = CL_SUCCESS;

    if (iterationsV > 0)
    {
        DeformerStatus dstatus;

        // set up openCL kernel
        SetUpKernel(block, numElements);

        if (averageKernel.get() != nullptr && tangentKernel.get() != nullptr)
        {
            // init data builds the neighbour table and we are going to upload it
            MObject referenceMeshV = block.inputValue(CustomDeltaMushDeformer::outputGeom).data();

            int size = numElements;
            m_size = size;
            neigh_table.resize(size * MAX_NEIGH);
            delta_table.resize(size * MAX_NEIGH);
            delta_size.resize(size);
            gpu_delta_table.resize(size * 3 * (MAX_NEIGH - 1));
            RebindData(referenceMeshV, iterationsV, amountV);

            // create buffers and upload data
            cl_int clStatus;

            d_neig_table.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                neigh_table.size() * sizeof(int), neigh_table.data(), &clStatus));
            MOpenCLInfo::checkCLErrorStatus(clStatus);

            d_delta_table.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                gpu_delta_table.size() * sizeof(float), gpu_delta_table.data(), &clStatus));
            MOpenCLInfo::checkCLErrorStatus(clStatus);

            d_primary = clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_READ_ONLY,
                MAX_NEIGH * size * sizeof(float), nullptr, &clStatus);
            MOpenCLInfo::checkCLErrorStatus(clStatus);

            d_secondary = clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_READ_ONLY,
                MAX_NEIGH * size * sizeof(float), nullptr, &clStatus);
            MOpenCLInfo::checkCLErrorStatus(clStatus);

            d_delta_size.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                delta_size.size() * sizeof(float), delta_size.data(), &clStatus));
            MOpenCLInfo::checkCLErrorStatus(clStatus);
        }

        // Set up our input events.  The input event could be NULL, in that case we need to pass
        // slightly different parameters into clEnqueueNDRangeKernel.
        cl_event events[1] = { 0 };
        cl_uint eventCount = 0;
        if (inputPositions.bufferReadyEvent().get())
        {
            events[eventCount++] = inputPositions.bufferReadyEvent().get();
        }

        void* src = (void*)&d_primary;
        void* trg = (void*)inputPositions.buffer().getReadOnlyRef();

        cl_event inEvent;
        cl_event outEvent;
        for (int i = 0; i < iterationsV; i++)
        {
            // Set all of our kernel parameters.
            // Input buffer and output buffer may be changing every frame, so always set them
            
            //swap(src, trg); // FIXME
            if (i==1)
            {
                trg = (void*)&d_secondary;
            }
            uint32_t parameterId = 0;
            err = clSetKernelArg(averageKernel.get(), parameterId++, sizeof(cl_mem), trg);
            MOpenCLInfo::checkCLErrorStatus(err);
            err = clSetKernelArg(averageKernel.get(), parameterId++, sizeof(cl_mem), d_neig_table.getReadOnlyRef());
            MOpenCLInfo::checkCLErrorStatus(err);
            err = clSetKernelArg(averageKernel.get(), parameterId++, sizeof(cl_mem), src);
            MOpenCLInfo::checkCLErrorStatus(err);
            err = clSetKernelArg(averageKernel.get(), parameterId++, sizeof(cl_float), (void*)&amountV);
            MOpenCLInfo::checkCLErrorStatus(err);
            err = clSetKernelArg(averageKernel.get(), parameterId++, sizeof(cl_uint), (void*)&i);
            MOpenCLInfo::checkCLErrorStatus(err);
            err = clSetKernelArg(averageKernel.get(), parameterId++, sizeof(cl_uint), (void*)&numElements);
            MOpenCLInfo::checkCLErrorStatus(err);

            // Run the Kernel
            if (i==0)
            {
                err = clEnqueueNDRangeKernel(
                    MOpenCLInfo::getMayaDefaultOpenCLCommandQueue(),
                    averageKernel.get(),
                    1,
                    nullptr,
                    &fGlobalWorkSize,
                    &fLocalWorkSize,
                    eventCount,
                    events,
                    &outEvent
                );
                inEvent = events[0];
            }
            else
            {
                err = clEnqueueNDRangeKernel(
                    MOpenCLInfo::getMayaDefaultOpenCLCommandQueue(),
                    averageKernel.get(),
                    1,
                    nullptr,
                    &fGlobalWorkSize,
                    &fLocalWorkSize,
                    1,
                    &inEvent,
                    &outEvent
                );
            }
            MOpenCLInfo::checkCLErrorStatus(err);

            cl_event tmp = outEvent;
            outEvent = inEvent;
            inEvent = tmp;
        }

        uint32_t parameterId = 0;
        err = clSetKernelArg(tangentKernel.get(), parameterId++, sizeof(cl_mem), outputPositions.buffer().getReadOnlyRef());
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangentKernel.get(), parameterId++, sizeof(cl_mem), d_delta_table.getReadOnlyRef());
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangentKernel.get(), parameterId++, sizeof(cl_mem), d_delta_size.getReadOnlyRef());
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangentKernel.get(), parameterId++, sizeof(cl_mem), d_neig_table.getReadOnlyRef());
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangentKernel.get(), parameterId++, sizeof(cl_mem), trg);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangentKernel.get(), parameterId++, sizeof(cl_uint), (void*)&numElements);
        MOpenCLInfo::checkCLErrorStatus(err);

        // Run the Kernel
        MAutoCLEvent tangentKernelFinishedEvent;
        err = clEnqueueNDRangeKernel(
            MOpenCLInfo::getMayaDefaultOpenCLCommandQueue(),
            tangentKernel.get(),
            1,
            nullptr,
            &fGlobalWorkSize,
            &fLocalWorkSize,
            0,
            nullptr,
            tangentKernelFinishedEvent.getReferenceForAssignment()
        );
        outputPositions.setBufferReadyEvent(tangentKernelFinishedEvent);
        MOpenCLInfo::checkCLErrorStatus(err);

        outputData.setBuffer(outputPositions);
        
        return kDeformerSuccess;
    }

    return kDeformerFailure;
}

void CustomDeltaMushGPUDeformer::terminate()
{
    cl_int err = CL_SUCCESS;

    d_neig_table.reset();
    d_delta_size.reset();
    d_delta_table.reset();

    MOpenCLInfo::releaseOpenCLKernel(averageKernel);
    averageKernel.reset();

    MOpenCLInfo::releaseOpenCLKernel(tangentKernel);
    tangentKernel.reset();
}

MPxGPUDeformer::DeformerStatus CustomDeltaMushGPUDeformer::SetUpKernel(MDataBlock& block, int32_t numElements)
{
    cl_int err = CL_SUCCESS;

    MString openCLKernelFile(CustomDeltaMushDeformer::pluginPath);
    openCLKernelFile += "/deltaMush.cl";
    MString openCLAverageKernelName("AverageOpencl");
    MString openCLTangentKernelName("TangentSpaceOpencl");
    averageKernel = MOpenCLInfo::getOpenCLKernel(openCLKernelFile, openCLAverageKernelName);
    tangentKernel = MOpenCLInfo::getOpenCLKernel(openCLKernelFile, openCLTangentKernelName);

    if (averageKernel.isNull())
    {
        MGlobal::displayError("error getting average kernel from file");
        return kDeformerFailure;
    }
    if (tangentKernel.isNull())
    {
        MGlobal::displayError("error getting tangent kernel from file");
        return kDeformerFailure;
    }

    // Figure out a good work group size for our kernel
    fLocalWorkSize = 0;
    fGlobalWorkSize = 0;
    size_t retSize = 0;
    err = clGetKernelWorkGroupInfo(
        averageKernel.get(),
        MOpenCLInfo::getOpenCLDeviceId(),
        CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(size_t),
        &fLocalWorkSize,
        &retSize
    );
    MOpenCLInfo::checkCLErrorStatus(err);
    if (err != CL_SUCCESS || retSize == 0 || fLocalWorkSize == 0)
    {
        return kDeformerFailure;
    }

    // Global work size must be a multiple of local work size
    const size_t remain = numElements % fLocalWorkSize;
    if (remain != 0)
    {
        fGlobalWorkSize = numElements + (fLocalWorkSize - remain);
    }
    else
    {
        fGlobalWorkSize = numElements;
    }

    return MPxGPUDeformer::DeformerStatus();
}

void CustomDeltaMushGPUDeformer::InitData(MObject& mesh)
{
}

void CustomDeltaMushGPUDeformer::ComputeDelta(MPointArray& source, MPointArray& target)
{
}

void CustomDeltaMushGPUDeformer::RebindData(MObject& mesh, int32_t iter, double amount)
{
}
