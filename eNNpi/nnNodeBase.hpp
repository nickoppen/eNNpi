#ifndef _nnNodeBase_h
#define _nnNodeBase_h

#include <sstream>


class nnNode
{
	public:
                            nnNode()
                            {
                                nodeValueIsSet = false;
                            }

                            nnNode(unsigned int newIndex)
                            {
                            // this does not get called for some reason
                            //	nodeValueIsSet = false;
                                index = newIndex;
                            }

	virtual					~nnNode() {};

    virtual	float			value()
                            {
                                return nodeValue;
                            }

    virtual bool			activationFromLink(float value) = 0;
    virtual float			adjustBiasReturningOutputError() = 0;

                            //	virtual	void			storeOn(sstring * strOut, unsigned int layerNo, unsigned int nodeNo) = 0;
//			void			setNodeName(sstring newName);

                            virtual	void			value(float newVal)
                            {
                                nodeValue = newVal;
                                nodeValueIsSet = true;
                            }

			unsigned int	nodeIndex() { return index; }

	// static training parameters
	public:
    static	void			setLearningParameters(float newLRP, float newMomentum)
                            {
                                learningRate = newLRP;
                                momentum = newMomentum;
                            }


	protected:
	static	float			learningRate;	
	static  float			momentum;

	// basic node member vars
	protected:
			float			nodeValue;
			unsigned int	index;
            string			nodeName;
//			HANDLE			syncEventHandle;	// run (in output and hidden nodes) and train (in hidden and input nodes)
//          QSemaphore      sem;

private:
			bool			nodeValueIsSet;
	
};

float nnNode::learningRate = (float)0.01;
float nnNode::momentum = (float)0.0;

#endif
