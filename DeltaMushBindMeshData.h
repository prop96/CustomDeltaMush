#pragma once
#include "DeltaMushUtil.h"
#include <maya/MVector.h>
#include <vector>

class DeltaMushBindMeshData
{
public:
	DeltaMushBindMeshData() = default;
	~DeltaMushBindMeshData() = default;

	void SetSmoothingData(uint32_t iter, double amount);
	const SmoothingData& GetSmoothingData() const;

	void SetBindMeshData(MObject& mesh);

	const std::vector<std::vector<int32_t>>& GetNeighbourIndices() const;
	const std::vector<float>& GetDeltaLength() const;
	const std::vector<std::vector<MVector>>& GetDelta() const;

	bool IsInitialized() const;

private:
	SmoothingData m_smoothingData;
	std::vector<std::vector<int32_t>> m_neighbourIndices;
	std::vector<float> m_deltaLength;
	std::vector<std::vector<MVector>> m_delta;
	bool m_isInitialized = false;

	void ComputeDelta(const std::vector<MPoint>& src, const std::vector<MPoint>& smoothed);
};