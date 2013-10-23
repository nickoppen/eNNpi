
#ifndef _unaryBiasNode_h
#define _unaryBiasNode_h

#include "nnNode.hpp"

class unaryBiasNode :   public inputNode
{
public:
	unaryBiasNode(network_description & net, unsigned int newIndex) : inputNode(net, newIndex) 
	{ 
		nodeName = "Bias Node";
		nodeValue = 1.0;
	};

	void	value(float newVal) { nodeValue = 1.0; };	// all other values are ignored
	float	value() { return 1.0; };

private:
	// nothing
};

#endif
