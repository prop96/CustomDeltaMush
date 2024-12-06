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

	m_neighbourIndices.clear();
	m_startIndexNeighbourIndices.resize(numVertices + 1);

	// ���_�̗אڏ����i�[
	for (iter.reset(); !iter.isDone(); iter.next())
	{
		const int32_t vIdx = iter.index();

		// �אڒ��_�C���f�b�N�X���擾
		MIntArray neighborIndices;
		iter.getConnectedVertices(neighborIndices);

		// private field �Ɋi�[
		const uint32_t numNeighbour = neighborIndices.length();
		for (uint32_t nIdx = 0; nIdx < numNeighbour; nIdx++)
		{
			m_neighbourIndices.push_back(neighborIndices[nIdx]);
		}
		m_startIndexNeighbourIndices[vIdx + 1] = m_startIndexNeighbourIndices[vIdx] + numNeighbour;
	}

	MFnMesh meshFn(mesh);

	// ���b�V���̒��_���W���擾
	std::vector<MPoint> posOriginal;
	{
		MPointArray tmpPosOriginal;
		meshFn.getPoints(tmpPosOriginal, MSpace::kObject);
		DMUtil::MPointArrayToVector(posOriginal, tmpPosOriginal);
	}

	// Smoothing ���� (posSmoothed ���v�Z)
	std::vector<MPoint> posSmoothed;
	DMUtil::ComputeSmoothedPoints(posOriginal, posSmoothed, m_smoothingData, m_startIndexNeighbourIndices, m_neighbourIndices);

	// Delta ���v�Z���� m_pointData �Ɋi�[
	ComputeDelta(posOriginal, posSmoothed);

	m_isInitialized = true;
}

const std::vector<uint32_t>& DeltaMushBindMeshData::GetStartIndexNeighbourIndices() const
{
	return m_startIndexNeighbourIndices;
}

const std::vector<int32_t>& DeltaMushBindMeshData::GetNeighbourIndices() const
{
	return m_neighbourIndices;
}

const std::vector<float>& DeltaMushBindMeshData::GetDeltaLength() const
{
	return m_deltaLength;
}

const std::vector<std::array<float, 3>>& DeltaMushBindMeshData::GetDelta() const
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
	m_delta.resize(m_startIndexNeighbourIndices[numVertices] - numVertices);
	m_deltaLength.resize(numVertices);

	// �e���_���ƂɃf���^���v�Z
	for (uint32_t vertIdx = 0; vertIdx < numVertices; vertIdx++)
	{
		// set delta length
		const MVector delta = MVector(src[vertIdx] - smoothed[vertIdx]);
		m_deltaLength[vertIdx] = delta.length();

		const uint32_t startIdx = m_startIndexNeighbourIndices[vertIdx];
		const uint32_t startIdxDelta = startIdx - vertIdx;

		// compute tangent matrix and delta in the tangent space
		const uint32_t numNeighbour = m_startIndexNeighbourIndices[vertIdx + 1] - startIdx;
		for (uint32_t neighborIdx = 0; neighborIdx < numNeighbour - 1; neighborIdx++)
		{
			const MMatrix mat = DMUtil::ComputeTangentMatrix(
				smoothed[vertIdx],
				smoothed[m_neighbourIndices[startIdx + neighborIdx]],
				smoothed[m_neighbourIndices[startIdx + neighborIdx + 1]]);

			// ���_�� tangent space coordinate �Ńf���^��ێ�����
			auto deltaTangentSpace = delta * mat.inverse();
			m_delta[startIdxDelta + neighborIdx][0] = deltaTangentSpace.x;
			m_delta[startIdxDelta + neighborIdx][1] = deltaTangentSpace.y;
			m_delta[startIdxDelta + neighborIdx][2] = deltaTangentSpace.z;
		}
	}
}
