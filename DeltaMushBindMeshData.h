#pragma once
#include "DeltaMushUtil.h"
#include <maya/MVector.h>
#include <vector>
#include <array>

class DeltaMushBindMeshData
{
public:
	DeltaMushBindMeshData() = default;
	~DeltaMushBindMeshData() = default;

	void SetSmoothingData(uint32_t iter, double amount);
	const SmoothingData& GetSmoothingData() const;

	void SetBindMeshData(MObject& mesh);

	const std::vector<int32_t>& GetNeighbourIndices() const;
	const std::vector<float>& GetDeltaLength() const;
	const std::vector<std::array<float, 3>>& GetDelta() const;
	const std::vector<uint32_t>& GetStartIndexNeighbourIndices() const;

	bool IsInitialized() const;

private:
	SmoothingData m_smoothingData;
	std::vector<int32_t> m_neighbourIndices;
	std::vector<float> m_deltaLength;
	std::vector<std::array<float, 3>> m_delta;
	bool m_isInitialized = false;

	std::vector<uint32_t> m_startIndexNeighbourIndices;

	void ComputeDelta(const std::vector<MPoint>& src, const std::vector<MPoint>& smoothed);
};