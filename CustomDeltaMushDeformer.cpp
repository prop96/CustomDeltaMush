#include "CustomDeltaMushDeformer.h"
#include "DeltaMushUtil.h"
#include <maya/MFnUnitAttribute.h>
#include <maya/MDistance.h>
#include <maya/MPoint.h>
#include <maya/MItMeshVertex.h>
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

	MMatrix ComputeTangentMatrix(const MPoint& pos, const MPoint& posNeighbor0, const MPoint& posNeighbor1)
	{
		// 注目している頂点と隣接頂点の作る三角形ポリゴンを考えて、tangent matrix を作る
		MVector v0 = posNeighbor0 - pos;
		MVector v1 = posNeighbor1 - pos;

		v0.normalize();
		v1.normalize();

		// tangent, normal, binormal
		MVector t = v0;
		MVector n = t ^ v1;
		MVector b = n ^ t;

		MMatrix mat = MMatrix();
		{
			mat[0][0] = t.x;
			mat[0][1] = t.y;
			mat[0][2] = t.z;
			mat[0][3] = 0;
			mat[1][0] = b.x;
			mat[1][1] = b.y;
			mat[1][2] = b.z;
			mat[1][3] = 0;
			mat[2][0] = n.x;
			mat[2][1] = n.y;
			mat[2][2] = n.z;
			mat[2][3] = 0;
			mat[3][0] = 0;
			mat[3][1] = 0;
			mat[3][2] = 0;
			mat[3][3] = 1;
		}

		return mat;
	}

	void MPointArrayToVector(std::vector<MPoint>& vec, const MPointArray& ptArr)
	{
		const uint32_t num = ptArr.length();
		vec.resize(num);
		for (uint32_t idx = 0; idx < num; idx++)
		{
			vec[idx] = ptArr[idx];
		}
	}

	void VectorToMPointArray(MPointArray& ptArr, const std::vector<MPoint>& vec)
	{
		const uint32_t num = vec.size();
		ptArr.setLength(num);
		for (uint32_t idx = 0; idx < num; idx++)
		{
			ptArr[idx] = vec[idx];
		}
	}
}

MStatus CustomDeltaMushDeformer::deform(MDataBlock& data, MItGeometry& iter, const MMatrix& localToWorld, unsigned int multiIdx)
{
	MStatus returnStat;

	// set smoothing property
	{
		int32_t iterationsVal = data.inputValue(smoothIterations).asInt();
		double amountVal = data.inputValue(smoothAmount).asDouble();
		SetSmoothingData(iterationsVal, amountVal);
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
	if (bool rebindVal = data.inputValue(rebind).asBool(); rebindVal || !m_isInitialized)
	{
		// bind the original mesh
		MObject originalGeomVal = data.inputArrayValue(origGeom, &returnStat).inputValue().asMesh();
		InitializeData(originalGeomVal);
		data.inputValue(rebind).setBool(false);
	}

	// noting to do if envelop is zero
	double envelopeVal = static_cast<double>(data.inputValue(envelope).asFloat());
	if (envelopeVal < Eps)
	{
		return MS::kSuccess;
	}

	// apply delta mush
	std::vector<MPoint> finalPos;
	{
		double applyDeltaVal = data.inputValue(applyDelta).asDouble();

		std::vector<MPoint> initPos;
		{
			MPointArray tmpInitPos;
			iter.allPositions(tmpInitPos, MSpace::kWorld);
			MPointArrayToVector(initPos, tmpInitPos);
		}

		ApplyDeltaMush(initPos, finalPos, envelopeVal, applyDeltaVal);
	}

	// setting all the points
	{
		MPointArray tmpFinalPos;
		VectorToMPointArray(tmpFinalPos, finalPos);
		iter.setAllPositions(tmpFinalPos);
	}

	return returnStat;
}

MStatus CustomDeltaMushDeformer::InitializeData(MObject& mesh)
{
	MStatus stat;

	// 頂点ごとのデータ配列を初期化
	m_pointData.clear();

	// 頂点の隣接情報を格納
	MItMeshVertex iter(mesh);
	for (iter.reset(); !iter.isDone(); iter.next())
	{
		PointData pd;

		// 隣接頂点インデックスを取得
		MIntArray neighborIndices;
		iter.getConnectedVertices(neighborIndices);
		const uint32_t neighbourNum = neighborIndices.length();

		// PointData に格納
		pd.NeighbourIndices.resize(neighbourNum);
		neighborIndices.get(pd.NeighbourIndices.data());

		// 隣接頂点ごとの delta を保持する配列を初期化
		pd.Delta.resize(neighbourNum - 1);

		m_pointData.push_back(pd);
	}

	MFnMesh meshFn(mesh);

	// メッシュの頂点座標を取得
	std::vector<MPoint> posOriginal;
	{
		MPointArray tmpPosOriginal;
		meshFn.getPoints(tmpPosOriginal, MSpace::kObject);
		MPointArrayToVector(posOriginal, tmpPosOriginal);
	}

	// Smoothing 処理 (posSmoothed を計算)
	std::vector<MPoint> posSmoothed;
	ComputeSmoothedPoints(posOriginal, posSmoothed);

	// Delta を計算して m_pointData に格納
	ComputeDelta(posOriginal, posSmoothed);

	m_isInitialized = true;

	return stat;
}

MStatus CustomDeltaMushDeformer::InitializeData(MObject& mesh, uint32_t smoothingIter, double smoothingAmount)
{
	SetSmoothingData(smoothingIter, smoothingAmount);
	return InitializeData(mesh);
}

void CustomDeltaMushDeformer::ApplyDeltaMush(const std::vector<MPoint>& skinned, std::vector<MPoint>& deformed, double envelope, double applyDeltaAmount) const
{
	// NOTE: skinned はワールド座標系での頂点位置と想定

	assert(m_isInitialized);

	const uint32_t numVerts = skinned.size();
	deformed.resize(numVerts);

	// compute mush
	std::vector<MPoint> mushed;
	ComputeSmoothedPoints(skinned, mushed);

	// apply delta to mush
	for (uint32_t vertIdx = 0; vertIdx < numVerts; vertIdx++)
	{
		const PointData& pointData = m_pointData[vertIdx];

		// compute delta in animated pose
		MVector delta = MVector::zero;

		// looping the neighbours
		const uint32_t neighbourNum = pointData.NeighbourIndices.size();
		for (uint32_t neighborIdx = 0; neighborIdx < neighbourNum - 1; neighborIdx++)
		{
			MMatrix mat = ComputeTangentMatrix(
				mushed[vertIdx],
				mushed[pointData.NeighbourIndices[neighborIdx]],
				mushed[pointData.NeighbourIndices[neighborIdx + 1]]);

			delta += (pointData.Delta[neighborIdx] * mat);
		}
		delta /= static_cast<double>(neighbourNum);

		// delta の長さを合わせる
		delta = delta.normal() * pointData.DeltaLength;

		// add delta to mush
		MPoint deltaMushed = mushed[vertIdx] + delta * applyDeltaAmount;

		// envelope を考慮
		deformed[vertIdx] = skinned[vertIdx] + envelope * (deltaMushed - skinned[vertIdx]);
	}
}

void CustomDeltaMushDeformer::SetSmoothingData(uint32_t iter, double amount)
{
	m_smoothingData.Iter = iter;
	m_smoothingData.Amount = amount;

	m_isInitialized = false;
}

void CustomDeltaMushDeformer::ComputeSmoothedPoints(const std::vector<MPoint>& src, std::vector<MPoint>& smoothed) const
{
#if 0
	// verify Laplacian Smoothing
	CHECK_MSTATUS(DMUtil::SmoothMesh(mesh, source, target, smoothItr, smoothAmount));
	return;
#endif

	const uint32_t numVerts = src.size();
	std::vector<MPoint> srcCopy(numVerts);

	smoothed.resize(numVerts);
	std::copy(src.begin(), src.end(), smoothed.begin());

	for (uint32_t itr = 0; itr < m_smoothingData.Iter; itr++)
	{
		srcCopy.swap(smoothed);

		for (uint32_t vertIdx = 0; vertIdx < numVerts; vertIdx++)
		{
			const PointData& pointData = m_pointData[vertIdx];

			// 隣接頂点の平均としてスムージング
			MVector smoothedPos = MVector::zero;
			for (const int neighbourIdx : pointData.NeighbourIndices)
			{
				smoothedPos += srcCopy[neighbourIdx];
			}
			smoothedPos *= 1.0 / double(pointData.NeighbourIndices.size());

			smoothed[vertIdx] = srcCopy[vertIdx] + (smoothedPos - srcCopy[vertIdx]) * m_smoothingData.Amount;
		}
	}
}

void CustomDeltaMushDeformer::ComputeDelta(const std::vector<MPoint>& src, const std::vector<MPoint>& smoothed)
{
	const uint32_t numVerts = src.size();

	// 各頂点ごとにデルタを計算
	for (uint32_t vertIdx = 0; vertIdx < numVerts; vertIdx++)
	{
		PointData& pointData = m_pointData[vertIdx];

		const MVector delta = MVector(src[vertIdx] - smoothed[vertIdx]);
		pointData.DeltaLength = delta.length();

		// compute tangent matrix and delta in the tangent space
		const uint32_t neighbourNum = pointData.NeighbourIndices.size();
		for (uint32_t neighborIdx = 0; neighborIdx < neighbourNum - 1; neighborIdx++)
		{
			MMatrix mat = ComputeTangentMatrix(
				smoothed[vertIdx],
				smoothed[pointData.NeighbourIndices[neighborIdx]],
				smoothed[pointData.NeighbourIndices[neighborIdx + 1]]);

			// 頂点の tangent space coordinate でデルタを保持する
			pointData.Delta[neighborIdx] = delta * mat.inverse();
		}
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
