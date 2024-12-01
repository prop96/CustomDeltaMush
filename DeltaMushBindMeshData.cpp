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

	MItMeshVertex iter(mesh);
	const uint32_t numVertices = iter.count();

	m_neighbourIndices.resize(numVertices);

	// 頂点の隣接情報を格納
	for (iter.reset(); !iter.isDone(); iter.next())
	{
		const int32_t idx = iter.index();

		// 隣接頂点インデックスを取得
		MIntArray neighborIndices;
		iter.getConnectedVertices(neighborIndices);
		const uint32_t numNeighbour = neighborIndices.length();

		// private field に格納
		m_neighbourIndices[idx].resize(numNeighbour);
		neighborIndices.get(m_neighbourIndices[idx].data());
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
	DMUtil::ComputeSmoothedPoints(posOriginal, posSmoothed, m_smoothingData, m_neighbourIndices);

	// Delta を計算して m_pointData に格納
	ComputeDelta(posOriginal, posSmoothed);

	m_isInitialized = true;

	//return stat;
}

const std::vector<std::vector<int32_t>>& DeltaMushBindMeshData::GetNeighbourIndices() const
{
	return m_neighbourIndices;
}

const std::vector<float>& DeltaMushBindMeshData::GetDeltaLength() const
{
	return m_deltaLength;
}

const std::vector<std::vector<std::array<float, 3>>>& DeltaMushBindMeshData::GetDelta() const
{
	return m_delta;
}

bool DeltaMushBindMeshData::IsInitialized() const
{
	return m_isInitialized;
}

void DeltaMushBindMeshData::ComputeDelta(const std::vector<MPoint>& src, const std::vector<MPoint>& smoothed)
{
	const uint32_t numVertices = src.size();
	m_delta.resize(numVertices);
	m_deltaLength.resize(numVertices);

	// 各頂点ごとにデルタを計算
	for (uint32_t vertIdx = 0; vertIdx < numVertices; vertIdx++)
	{
		const MVector delta = MVector(src[vertIdx] - smoothed[vertIdx]);
		m_deltaLength[vertIdx] = delta.length();

		// 隣接頂点ごとの delta を保持する配列を初期化
		const uint32_t numNeighbour = m_neighbourIndices[vertIdx].size();
		m_delta[vertIdx].resize(numNeighbour - 1);

		// compute tangent matrix and delta in the tangent space
		for (uint32_t neighborIdx = 0; neighborIdx < numNeighbour - 1; neighborIdx++)
		{
			MMatrix mat = DMUtil::ComputeTangentMatrix(
				smoothed[vertIdx],
				smoothed[m_neighbourIndices[vertIdx][neighborIdx]],
				smoothed[m_neighbourIndices[vertIdx][neighborIdx + 1]]);

			// 頂点の tangent space coordinate でデルタを保持する
			auto deltaTangentSpace = delta * mat.inverse();
			m_delta[vertIdx][neighborIdx][0] = deltaTangentSpace.x;
			m_delta[vertIdx][neighborIdx][1] = deltaTangentSpace.y;
			m_delta[vertIdx][neighborIdx][2] = deltaTangentSpace.z;
		}
	}
}
