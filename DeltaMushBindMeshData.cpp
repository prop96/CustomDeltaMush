#include "DeltaMushBindMeshData.h"
#include <maya/MItMeshVertex.h>
#include <maya/MIntArray.h>
#include <maya/MPointArray.h>
#include <maya/MFnMesh.h>


void DeltaMushBindMeshData::SetSmoothingData(uint32_t iter, double amount)
{
	m_smoothingData.Iter = iter;
	m_smoothingData.Amount = amount;

	m_isInitialized = false;
}

const SmoothingData& DeltaMushBindMeshData::GetSmoothingData() const
{
	return m_smoothingData;
}

void DeltaMushBindMeshData::SetBindMeshData(MObject& mesh)
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
		DMUtil::MPointArrayToVector(posOriginal, tmpPosOriginal);
	}

	// Smoothing 処理 (posSmoothed を計算)
	std::vector<MPoint> posSmoothed;
	DMUtil::ComputeSmoothedPoints(posOriginal, posSmoothed, m_smoothingData, m_pointData);

	// Delta を計算して m_pointData に格納
	ComputeDelta(posOriginal, posSmoothed);

	m_isInitialized = true;

	//return stat;
}

const std::vector<PointData>& DeltaMushBindMeshData::GetPointData() const
{
	return m_pointData;
}

bool DeltaMushBindMeshData::IsInitialized() const
{
	return m_isInitialized;
}

void DeltaMushBindMeshData::ComputeDelta(const std::vector<MPoint>& src, const std::vector<MPoint>& smoothed)
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
			MMatrix mat = DMUtil::ComputeTangentMatrix(
				smoothed[vertIdx],
				smoothed[pointData.NeighbourIndices[neighborIdx]],
				smoothed[pointData.NeighbourIndices[neighborIdx + 1]]);

			// 頂点の tangent space coordinate でデルタを保持する
			pointData.Delta[neighborIdx] = delta * mat.inverse();
		}
	}
}
