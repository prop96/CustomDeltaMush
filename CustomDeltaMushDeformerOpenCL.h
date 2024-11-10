#pragma once

#include <maya/MPxGPUDeformer.h>
#include <maya/MGPUDeformerRegistry.h>
#include <maya/MDataBlock.h>
#include <maya/MOpenCLInfo.h>
#include <maya/MPointArray.h>
#include <clew/clew_cl.h>
#include <vector>


// The GPU override implementation of the identityNode
class CustomDeltaMushGPUDeformer : public MPxGPUDeformer
{
public:
    static MGPUDeformerRegistrationInfo* getGPUDeformerInfo();
    static bool validateNodeInGraph(MDataBlock& block, const MEvaluationNode&, const MPlug& plug, MStringArray* messages);
    static bool validateNodeValues(MDataBlock& block, const MEvaluationNode&, const MPlug& plug, MStringArray* messages);

    inline static const MString nodeClassName = "customDeltaMushGPU";

    // Virtual methods from MPxGPUDeformer
    CustomDeltaMushGPUDeformer();
    virtual ~CustomDeltaMushGPUDeformer();
    static void* creator();
    static MStatus initialize();

    // Implementation of MPxGPUDeformer.
    MPxGPUDeformer::DeformerStatus evaluate(
        MDataBlock& block,
        const MEvaluationNode& evaluationNode,
        const MPlug& outputPlug,
        const MPlugArray& inputPlugs,
        const MGPUDeformerData& inputData,
        MGPUDeformerData& outputData) override;
    void terminate() override;

    MPxGPUDeformer::DeformerStatus SetUpKernel(MDataBlock& block, int32_t numElements);
    void InitData(MObject& mesh);
    void ComputeDelta(MPointArray& source, MPointArray& target);
    void RebindData(MObject& mesh, int32_t iter, double amount);

private:
    // Kernel
    MAutoCLKernel averageKernel;
    MAutoCLKernel tangentKernel;
    size_t fLocalWorkSize = 0;
    size_t fGlobalWorkSize = 0;
    int32_t m_size;

    cl_mem d_primary;
    cl_mem d_secondary;
    MAutoCLMem d_neig_table;
    MAutoCLMem d_delta_size;
    MAutoCLMem d_delta_table;

    std::vector<int32_t> neigh_table;
    std::vector<MVector> delta_table;
    std::vector<float> gpu_delta_table;
    std::vector<float> delta_size;
};

// registration information for the GPU deformer.
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
