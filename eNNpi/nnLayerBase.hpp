#ifndef _nnLayerBase_h
#define _nnLayerBase_h

#include <sstream>
#include <vector>
#include <random>
#include "twoDFloatArray.hpp"
#include "nnNodeBase.hpp"
#include "unaryBiasNode.hpp"
#include "nodeInputType.hpp"

class nnLayer
{
	public:
                        nnLayer(unsigned int layerIndex)
                        {
                            //nodes = new BList(1);
                            index = layerIndex;
                        }

	virtual				~nnLayer() { };
	
//	virtual void		storeOn(sstring * strOut, unsigned int layerNo) = 0;	// Store the layer on the given stream or database

	virtual	void		setLinkWeights(twoDFloatArray * weightArray) = 0;
	virtual	void		setNodeBiases(vector<float> * nodeArray) = 0;
    virtual int			randomise(random_device & randSeed) = 0;
				
	protected:
		unsigned int	index;
	
	private:

	
};

class inLayer : public nnLayer
{
	public:
		inLayer(network_description & net, unsigned int layerIndex) : nnLayer(layerIndex)
		{
			if (net.hasInputLayerBiasNode())
			{
				// the biasNode is created and stored in the inputlayer constructor
				hasBias = true;
//				cout << "setting inNode::hasBias = true\n";
			}
			else
			{
				hasBias = false;
//				cout<<"setting inNode::hasBias = false\n";
			}
		}

		virtual				~inLayer() {  }
		bool				hasBiasNode() { return hasBias; }

		virtual void		storeOn(stringstream * strOut)
		{
			if (hasBias)
				(*strOut) << "layerModifier(0,biasNode:true)\n";
			else
				(*strOut) << "layerModifier(0,biasNode:false)\n";
		}

	protected:
		unaryBiasNode *		biasNode;

	private:
		bool				hasBias;

};
#endif
