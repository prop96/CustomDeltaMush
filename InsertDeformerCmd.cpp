#include "InsertDeformerCmd.h"
#include "CustomDeltaMushDeformer.h"
#include "PrecomputeCustomSkinningNode.h"
#include <maya/MSelectionList.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnSkinCluster.h>
#include <maya/MPxSkinCluster.h>
#include <maya/MItSelectionList.h>
#include <maya/MItDependencyGraph.h>
#include <maya/MPlug.h>
#include <cassert>


void* InsertDeformerCmd::creator()
{
	return new InsertDeformerCmd();
}

MStatus InsertDeformerCmd::doIt(const MArgList&)
{
	return redoIt();
}

MStatus InsertDeformerCmd::undoIt()
{
	return m_dgMod.undoIt();
}

MStatus InsertDeformerCmd::redoIt()
{
	return InsertNode();
}

MStatus InsertDeformerCmd::InsertNode()
{
	MStatus returnStat;

	// get selected objects
	MSelectionList selection;
	CHECK_MSTATUS(MGlobal::getActiveSelectionList(selection));
	const unsigned int numSelection = selection.length(&returnStat);
	if (numSelection == 0)
	{
		return returnStat;
	}

	// find SkinCluster nodes of the selected objects
	MItSelectionList iter(selection);
	for (iter.reset(); !iter.isDone(); iter.next())
	{
		// get DAG path of the mesh node
		MDagPath dagPath;
		CHECK_MSTATUS(iter.getDagPath(dagPath));
		CHECK_MSTATUS(dagPath.extendToShape());

		// get the current SkinCluster node
		MObject skclObj = FindSkinClusterNode(dagPath, &returnStat);
		CHECK_MSTATUS(returnStat);
		if (skclObj.isNull())
		{
			continue;
		}
		MFnSkinCluster skclFn(skclObj);

		// create the new node to insert
		MObject precomputer = m_dgMod.createNode(PrecomputeCustomSkinningNode::nodeTypeName, &returnStat);
		CHECK_MSTATUS(returnStat);
		MFnDependencyNode precompFn(precomputer);
		MObject newDeformer = m_dgMod.createNode(CustomDeltaMushDeformer::nodeTypeName, &returnStat);
		CHECK_MSTATUS(returnStat);
		MFnDependencyNode deformerFn(newDeformer);

		// connect
		CHECK_MSTATUS(ConnectSameAttribute("deltaMushMatrix", precompFn, deformerFn));
		CHECK_MSTATUS(ConnectSameAttribute("weightList", skclFn, deformerFn));
		CHECK_MSTATUS(ConnectSameAttribute("originalGeometry[0]", skclFn, deformerFn));
		CHECK_MSTATUS(ReplaceConnection("outputGeometry[0]", skclFn, deformerFn, false));
		CHECK_MSTATUS(ConnectAttribute("originalGeometry[0]", skclFn, "originalGeometry", precompFn));
		CHECK_MSTATUS(ConnectAttribute("outputGeometry[0]", skclFn, "input[0].inputGeometry", deformerFn));
	}

	return m_dgMod.doIt();
}

MObject InsertDeformerCmd::FindSkinClusterNode(const MDagPath& meshPath, MStatus* ptrStat)
{
	MFnDagNode dagNode(meshPath);

	// mesh node must have inMesh attribute
	MPlug inMeshPlug = dagNode.findPlug("inMesh", ptrStat);
	CHECK_MSTATUS(*ptrStat);
	if (!inMeshPlug.isConnected(ptrStat))
	{
		return MObject::kNullObj;
	}
	CHECK_MSTATUS(*ptrStat);

	// search skin cluster node, upwards from the mesh node
	MItDependencyGraph dgIt(
		inMeshPlug,
		MFn::kInvalid,
		MItDependencyGraph::kUpstream,
		MItDependencyGraph::kDepthFirst,
		MItDependencyGraph::kPlugLevel,
		ptrStat);
	CHECK_MSTATUS(*ptrStat);

	for (dgIt.disablePruningOnFilter(); !dgIt.isDone(); dgIt.next())
	{
		// return if SkinCluster or CustomSkinCluster node
		MObject thisNode = dgIt.thisNode();
		if (thisNode.apiType() == MFn::kSkinClusterFilter || thisNode.apiType() == MFn::kPluginSkinCluster)
		{
			return thisNode;
		}
	}

	return MObject::kNullObj;
}

MPlug FindPlug(const MString& attrName, const MFnDependencyNode& node, MStatus* ptrStat)
{
	// split the attribute name by '.'
	MStringArray attrPath;
	CHECK_MSTATUS(attrName.split('.', attrPath));

	// parse the attribute path and obtain the instance of the plug
	MPlug plug;
	for (auto it = attrPath.begin(); it != attrPath.end(); ++it)
	{
		// query if the string includes any parenthesis
		MStringArray strArray;
		CHECK_MSTATUS((*it).split('[', strArray));

		// don't assume the dual parenthesis such as "hoge[0][0]"
		assert(strArray.length() == 1 || strArray.length() == 2);

		// anyway, the first string represents the attribute name
		MObject obj = node.attribute(strArray[0], ptrStat);
		CHECK_MSTATUS(*ptrStat);

		// get the plug instance from MObject
		if (it == attrPath.begin())
		{
			plug = node.findPlug(obj, ptrStat);
			CHECK_MSTATUS(*ptrStat);
		}
		else
		{
			plug = plug.child(obj, ptrStat);
			CHECK_MSTATUS(*ptrStat);
		}

		// if the string includes a parenthesis, get the corresponding element
		if (strArray.length() == 2)
		{
			CHECK_MSTATUS(strArray[1].split(']', strArray));
			unsigned int val = strArray[0].asUnsigned();
			plug = plug.elementByLogicalIndex(val, ptrStat);
			CHECK_MSTATUS(*ptrStat);
		}
	}

	return plug;
}

MStatus InsertDeformerCmd::ReplaceConnection(const MString& attrName, const MFnDependencyNode& src, const MFnDependencyNode& dst, bool asDst)
{
	MStatus returnStatus;

	// parse the attribute path and obtain the instance of the plug
	MPlug srcPlug = FindPlug(attrName, src, &returnStatus);
	MPlug dstPlug = FindPlug(attrName, dst, &returnStatus);

	// search plugs connecting to the given attribute in the src SkinCluster
	MPlugArray connectedPlugs;
	srcPlug.connectedTo(connectedPlugs, asDst, !asDst, &returnStatus);
	CHECK_MSTATUS(returnStatus);

	// connect the given attribute in the dst SkinCluster with the found plug
	for (const auto& plug : connectedPlugs)
	{
		if (asDst)
		{
			CHECK_MSTATUS(m_dgMod.disconnect(plug, srcPlug));
			CHECK_MSTATUS(m_dgMod.connect(plug, dstPlug));
		}
		else
		{
			CHECK_MSTATUS(m_dgMod.disconnect(srcPlug, plug));
			CHECK_MSTATUS(m_dgMod.connect(dstPlug, plug));
		}
	}

	return returnStatus;
}

MStatus InsertDeformerCmd::ConnectSameAttribute(const MString& attrName, const MFnDependencyNode& src, const MFnDependencyNode& dst)
{
	return ConnectAttribute(attrName, src, attrName, dst);
}

MStatus InsertDeformerCmd::ConnectAttribute(const MString& srcAttrName, const MFnDependencyNode& src, const MString& dstAttrName, const MFnDependencyNode& dst)
{
	MStatus returnStatus;

	MPlug srcPlug = FindPlug(srcAttrName, src, &returnStatus);
	CHECK_MSTATUS(returnStatus);
	MPlug dstPlug = FindPlug(dstAttrName, dst, &returnStatus);
	CHECK_MSTATUS(returnStatus);
	CHECK_MSTATUS(m_dgMod.connect(srcPlug, dstPlug));

	return returnStatus;
}
