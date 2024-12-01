#include "CustomDeltaMushDeformer.h"
#include "DeltaMushUtil.h"
#include <maya/MFnUnitAttribute.h>
#include <maya/MDistance.h>
#include <maya/MPoint.h>
#include <maya/MItMeshVertex.h>
#include <maya/MItGeometry.h>
#include <maya/MPointArray.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMatrixArrayData.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MGlobal.h>
#include <cassert>
#include <numeric>

// instancing the static fields
MTypeId CustomDeltaMushDeformer::id(0x80095);
MString CustomDeltaMushDeformer::pluginPath;
MObject CustomDeltaMushDeformer::rebind;
MObject CustomDeltaMushDeformer::smoothIterations;
MObject CustomDeltaMushDeformer::applyDelta;
MObject CustomDeltaMushDeformer::smoothAmount;

namespace {
	constexpr double Eps = std::numeric_limits<double>::epsilon();
}

MStatus CustomDeltaMushDeformer::deform(MDataBlock& data, MItGeometry& iter, const MMatrix& localToWorld, unsigned int multiIdx)
{
	MStatus returnStat;

	// set smoothing property
	{
		const int32_t iterationsVal = data.inputValue(smoothIterations).asInt();
		const double amountVal = data.inputValue(smoothAmount).asDouble();
		m_bindMeshData.SetSmoothingData(iterationsVal, amountVal);
	}

	// get originalGeometry attribute
	MFnDependencyNode thisNode(thisMObject());
	MObject origGeom = thisNode.attribute("originalGeometry", &returnStat);
	CHECK_MSTATUS(returnStat);

	// error if originalGeometry attribute is not connected
	if (MPlug refMeshPlug(thisMObject(), origGeom); !refMeshPlug.elementByLogicalIndex(0).isConnected(&returnStat))
	{
		returnStat.perror("mesh to bind is not connected");
		return MS::kFailure;
	}

	// initialize data for the mesh to bind if still not
	if (bool rebindVal = data.inputValue(rebind).asBool(); rebindVal || !m_bindMeshData.IsInitialized())
	{
		// bind the original mesh
		MObject originalGeomVal = data.inputArrayValue(origGeom, &returnStat).inputValue().asMesh();
		m_bindMeshData.SetBindMeshData(originalGeomVal);
		data.inputValue(rebind).setBool(false);
	}

	// noting to do if envelop is zero
	const double envelopeVal = static_cast<double>(data.inputValue(envelope).asFloat());
	if (envelopeVal < Eps)
	{
		return MS::kSuccess;
	}

	// apply delta mush
	std::vector<MPoint> finalPos;
	{
		const double applyDeltaVal = data.inputValue(applyDelta).asDouble();

		std::vector<MPoint> initPos;
		{
			MPointArray tmpInitPos;
			iter.allPositions(tmpInitPos, MSpace::kWorld);
			DMUtil::MPointArrayToVector(initPos, tmpInitPos);
		}

		ApplyDeltaMush(initPos, finalPos, envelopeVal, applyDeltaVal);
	}

	// setting all the points
	{
		MPointArray tmpFinalPos;
		DMUtil::VectorToMPointArray(tmpFinalPos, finalPos);
		iter.setAllPositions(tmpFinalPos);
	}

	return returnStat;
}

void CustomDeltaMushDeformer::ApplyDeltaMush(const std::vector<MPoint>& skinned, std::vector<MPoint>& deformed, double envelope, double applyDeltaAmount) const
{
	// NOTE: skinned はワールド座標系での頂点位置と想定

	assert(m_bindMeshData.IsInitialized());

	const uint32_t numVerts = skinned.size();
	deformed.resize(numVerts);

	auto& neighbourIndicesAll = m_bindMeshData.GetNeighbourIndices();
	auto& deltaLengthAll = m_bindMeshData.GetDeltaLength();
	auto& deltaAll = m_bindMeshData.GetDelta();

	// compute mush
	std::vector<MPoint> mushed;
	DMUtil::ComputeSmoothedPoints(skinned, mushed, m_bindMeshData.GetSmoothingData(), neighbourIndicesAll);

	// apply delta to mush
	for (uint32_t vertIdx = 0; vertIdx < numVerts; vertIdx++)
	{
		// compute delta in animated pose
		MVector delta = MVector::zero;

		auto& neighbourIndices = neighbourIndicesAll[vertIdx];

		// looping the neighbours
		const uint32_t neighbourNum = neighbourIndices.size();
		for (uint32_t neighborIdx = 0; neighborIdx < neighbourNum - 1; neighborIdx++)
		{
			MMatrix mat = DMUtil::ComputeTangentMatrix(
				mushed[vertIdx],
				mushed[neighbourIndices[neighborIdx]],
				mushed[neighbourIndices[neighborIdx + 1]]);

			delta += MVector(deltaAll[vertIdx][neighborIdx].data()) * mat;
		}
		delta /= static_cast<double>(neighbourNum);

		// delta の長さを合わせる
		delta = delta.normal() * deltaLengthAll[vertIdx];

		// add delta to mush
		MPoint deltaMushed = mushed[vertIdx] + delta * applyDeltaAmount;

		// envelope を考慮
		deformed[vertIdx] = skinned[vertIdx] + envelope * (deltaMushed - skinned[vertIdx]);
	}
}

void* CustomDeltaMushDeformer::creator()
{
	return new CustomDeltaMushDeformer();
}

MStatus CustomDeltaMushDeformer::initialize()
{
	MStatus returnStat;

	MFnTypedAttribute tAttr;
	MFnNumericAttribute nAttr;

	rebind = nAttr.create("rebind", "rbn", MFnNumericData::kBoolean, 1, &returnStat);
	CHECK_MSTATUS(nAttr.setKeyable(true));
	CHECK_MSTATUS(nAttr.setStorable(true));
	CHECK_MSTATUS(addAttribute(rebind));

	applyDelta = nAttr.create("applyDelta", "apdlt", MFnNumericData::kDouble, 1.0, &returnStat);
	CHECK_MSTATUS(nAttr.setKeyable(true));
	CHECK_MSTATUS(nAttr.setStorable(true));
	CHECK_MSTATUS(nAttr.setMin(0));
	CHECK_MSTATUS(nAttr.setMax(1));
	CHECK_MSTATUS(addAttribute(applyDelta));

	smoothIterations = nAttr.create("smoothIterations", "itr", MFnNumericData::kInt, 0, &returnStat);
	CHECK_MSTATUS(nAttr.setKeyable(true));
	CHECK_MSTATUS(nAttr.setStorable(true));
	CHECK_MSTATUS(nAttr.setMin(0));
	CHECK_MSTATUS(addAttribute(smoothIterations));

	smoothAmount = nAttr.create("smoothAmount", "sa", MFnNumericData::kDouble, 0.5, &returnStat);
	CHECK_MSTATUS(nAttr.setKeyable(true));
	CHECK_MSTATUS(nAttr.setStorable(true));
	CHECK_MSTATUS(nAttr.setMin(0));
	CHECK_MSTATUS(nAttr.setMax(1));
	CHECK_MSTATUS(addAttribute(smoothAmount));

	CHECK_MSTATUS(attributeAffects(rebind, outputGeom));
	CHECK_MSTATUS(attributeAffects(applyDelta, outputGeom));
	CHECK_MSTATUS(attributeAffects(smoothIterations, outputGeom));
	CHECK_MSTATUS(attributeAffects(smoothAmount, outputGeom));

	//MGlobal::executeCommand("makePaintable -attrType multiFloat -sm deformer CustomDeltaMushDeformer weights");

	return returnStat;
}
