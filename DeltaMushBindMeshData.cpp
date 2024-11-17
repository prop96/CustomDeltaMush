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

	// ���_���Ƃ̃f�[�^�z���������
	m_pointData.clear();

	// ���_�̗אڏ����i�[
	MItMeshVertex iter(mesh);
	for (iter.reset(); !iter.isDone(); iter.next())
	{
		PointData pd;

		// �אڒ��_�C���f�b�N�X���擾
		MIntArray neighborIndices;
		iter.getConnectedVertices(neighborIndices);
		const uint32_t neighbourNum = neighborIndices.length();

		// PointData �Ɋi�[
		pd.NeighbourIndices.resize(neighbourNum);
		neighborIndices.get(pd.NeighbourIndices.data());

		// �אڒ��_���Ƃ� delta ��ێ�����z���������
		pd.Delta.resize(neighbourNum - 1);

		m_pointData.push_back(pd);
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
	DMUtil::ComputeSmoothedPoints(posOriginal, posSmoothed, m_smoothingData, m_pointData);

	// Delta ���v�Z���� m_pointData �Ɋi�[
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

	// �e���_���ƂɃf���^���v�Z
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

			// ���_�� tangent space coordinate �Ńf���^��ێ�����
			pointData.Delta[neighborIdx] = delta * mat.inverse();
		}
	}
}
