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

	void SetPointData(MObject& mesh);
	const std::vector<PointData>& GetPointData() const;

	bool IsInitialized() const;

private:
	std::vector<PointData> m_pointData;
	SmoothingData m_smoothingData;
	bool m_isInitialized = false;

	void ComputeDelta(const std::vector<MPoint>& src, const std::vector<MPoint>& smoothed);
};