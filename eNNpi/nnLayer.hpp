#ifndef _nnLayer_h
#define _nnLayer_h

#include <sstream>
#include "networkDescription.hpp"
#include "nnLayerBase.hpp"
#include "nnNode.hpp"
#include "errStruct.hpp"

// C declarations
/*
unsigned int runner(void * arg)             // thread proc for input node to commence run
{
    Sleep(0);
    ((inputNode * )arg)->pushValue();
    return 1;
}

unsigned int trainerI(void * arg)			// thread proc for input node to commence train cycle
{
    Sleep(0);
    if (((inputNode*)arg)->train() == 0)
        throw internal_Error(ENN_ERR_TRAIN_TRAIN);
    return 1;
}

unsigned int trainerH(void * arg)			// thread proc for hidden node to commence train cycle
{
    Sleep(0);
    if (((hiddenNode*)arg)->train() == 0)
        throw internal_Error(ENN_ERR_TRAIN_TRAIN);
    return 1;
}
*/

class inputLayer : public inLayer
{
	// setup
	public:
                                            inputLayer(network_description & net, unsigned int layerIndex) : inLayer(net, layerIndex)
                                            {
                                                unsigned int i;
                                                unsigned int vectorSize;

                                                // create the list of nodes
                                                vectorSize = net.standardInputNodes();

                                                nodes = new vector<inputNode*> (net.inputNodes());
                                                stdNodes = new vector<inputNode*> (vectorSize);

                                                for (i = 0; i < vectorSize; i++)
                                                {
                                                    nodes->operator[](i) = new inputNode(net, i);		// refer to the same input nodes in both vectors
                                                    stdNodes->operator[](i) = nodes->operator[](i);
                                                }

                                                if (inLayer::hasBiasNode())
                                                {
                                                    inLayer::biasNode = new unaryBiasNode(net, i);
                                                    nodes->operator[](i) = ((inputNode*)inLayer::biasNode);
                                                }
                                            }

    virtual									~inputLayer()
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    delete (*nodeI);

                                                delete nodes;
                                                delete stdNodes;	// the content is shared with nodes
                                            }

    virtual void							setLinkWeights(twoDFloatArray * weightArray)
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                    (*nodeI)->setLinkWeights(weightArray->values((*nodeI)->nodeIndex()));
                                                }
                                            }

    virtual	void							setNodeBiases(vector<float> * nodeArray) { /* input layers have no biases */ }

            void							connectNodes(vector<hiddenNode*> * outputNodes)
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    (*nodeI)->connectTo(outputNodes);
                                            }


	// Access
    virtual void							storeOn(stringstream * strOut)
                                            {
                                                inLayer::storeOn(strOut);
                                                int n = 0;

                                                (*strOut) << "comment(Storing the input layer with: " << nodes->size() << " nodes.)\n";
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                	(*strOut) << "comment(" << n++ <<" Input Layer - Storing index: " << nnLayer::index << " node: " << (*nodeI)->nodeIndex() << ")\n";
                                                    (*nodeI)->storeOn(strOut, nnLayer::index);
                                                }
                                            }

            vector<inputNode*>			*	nodeList()
                                            {
                                                return nodes;
                                            }

			
	// save & retrieve
	public:
            void							linkWeights(twoDFloatArray * weightArray) {}
            void							nodeBiases(vector<float> * nodeArray) {}
	
	// run (and train)
	public:
            void							setInputVector(vector<float> * inVals)	// Load the input nodes with the given values
                                            {
                                                unsigned int stdNodeCount = standardNodes();

                                                if (inVals->size() == stdNodeCount)
                                                    for (nodeI = stdNodes->begin(); nodeI != stdNodes->end(); nodeI++)
                                                        (*nodeI)->value(inVals->operator[]((*nodeI)->nodeIndex()));
                                                else
                                                    throw; // do something
                                            }

//			thread_id						run()								// run the network and return the last thread---------------------------------------------------------------
            void							run()
                                            {
//                                                thread_id trId = 0;

                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                            #ifdef MULTI_THREADED
                                                    // multi-threading version
                                                    // runner calls pushValue for each input node
                                                    AfxBeginThread(runner, ((LPVOID)(*nodeI)), THREAD_PRIORITY_BELOW_NORMAL);
                                            #else
                                                    // sigle threading version
                                                    // call pushValue for each inputnode
                                                    (*nodeI)->pushValue();
                                            #endif
                                                }

//                                                return trId;	// return the last thread spawned (0 if no threads were spawned)
                                            }

    virtual int								randomise(random_device & randSeed) { return -1; }

            void							train()
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                            #ifdef MULTI_THREADED
                                                    // multi-threading version
                                                    // trainer calls train() for each hidden node
                                                    AfxBeginThread(trainerI, ((LPVOID)(*nodeI)), THREAD_PRIORITY_BELOW_NORMAL);
                                            #else
                                                    // single threading version
                                                    (*nodeI)->train();
                                            #endif
                                                }
                                            }

			status_t						waitForTraining()
                                            {
                                                status_t endValue;

                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                    endValue = (*nodeI)->waitForTraining();
                                                    if (endValue == 0)
                                                        throw internal_Error(ENN_ERR_TRAIN_WAITFORTRAIN);
                                                }
                                                return endValue;
                                            }

            void							blockTillTrained()
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++) // wait till all values are in using CEvent
                                                    (*nodeI)->blockTillTrained();

                                            //	for (i = 0; i < nodes->size(); i++) // release all semaphores
                                            //		(nodes->operator[](i))->mainThreadRelease();
                                            }


	private:
			vector<inputNode*>			*	nodes;									// all nodes including unary Bias node
			vector<inputNode*>			*	stdNodes;								// just the standard input nodes
			vector<inputNode*>::iterator	nodeI;									// general purpose iterator used all over the place

            unsigned int					standardNodes()
                                            {
                                                size_t stdNodeCount;

                                                if (inLayer::hasBiasNode())
                                                    stdNodeCount = nodes->size() - 1;
                                                else
                                                    stdNodeCount = nodes->size();

                                                return (unsigned int)stdNodeCount;
                                            }

};

class hiddenLayer : public inLayer
{
	// setup
	public:
                                            hiddenLayer(network_description & net, unsigned int layerIndex) : inLayer(net, layerIndex)
                                            {
                                                unsigned int i;
                                                learningRate = net.trainingLearningRate();
                                                // create the list of nodes
                                                nodes = new vector<hiddenNode*> (net.hiddenNodes());
                                                for (i = 0; i < net.hiddenNodes(); i++)
                                                    nodes->operator[](i) = new hiddenNode(net, i);
                                            }

    virtual									~hiddenLayer()
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    delete (*nodeI);

                                                delete nodes;
                                            }

    virtual void							setLinkWeights(twoDFloatArray * weightArray)
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    (*nodeI)->setLinkWeights(weightArray->values((*nodeI)->nodeIndex()));
                                            }

    virtual	void							setNodeBiases(vector<float> * nodeBiasArray)
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                    (*nodeI)->setBias(nodeBiasArray->operator[]((*nodeI)->nodeIndex()));
                                                }
                                            }


	vector<hiddenNode*>*					nodeList() { return nodes; }
								
            void							connectNodes(vector<outputNode*> * outputNodes)
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    (*nodeI)->connectTo(outputNodes);
                                            }


	private:
			vector<hiddenNode*> *			nodes;
			vector<hiddenNode*>::iterator	nodeI;
		
	// save & retrieve
	public:
            void							storeOn(stringstream * strOut)
                                            {
                                                (*strOut) << "comment(Storing the hidden layer)\n";

                                                // store any hidden layer modifiers here

                                                // store all the nodes
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                    (*nodeI)->storeOn(strOut, nnLayer::index);
                                                }
                                            }

			void							linkWeights(twoDFloatArray * weightArray);
			void							nodeBiases(vector<float> * nodeArray);	
	
	// training
	public:
            void							train()
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                            #ifdef MULTI_THREADED
                                                    // multi-threading version
                                                    // trainer calls train() for each hidden node
                                                    AfxBeginThread(trainerH, ((LPVOID)(*nodeI)), THREAD_PRIORITY_BELOW_NORMAL);
                                            #else
                                                    // single threading version
                                                    (*nodeI)->train();
                                            #endif
                                                }
                                            }

			status_t						waitForTraining()
                                            {
                                                status_t endValue;

                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                    endValue = (*nodeI)->waitForTraining();
                                                    if (endValue == 0)
                                                        throw internal_Error(ENN_ERR_TRAIN_WAITFORTRAIN);
                                                }
                                                return endValue;
                                            }

            void							blockTillTrained()
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++) // wait till all values are in using CEvent
                                                    (*nodeI)->blockTillTrained();

                                            //	for (i = 0; i < nodes->size(); i++) // release all semaphores
                                            //		(nodes->operator[](i))->mainThreadRelease();
                                            }

            int								randomise(random_device & randSeed)
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    (*nodeI)->randomise(randSeed, node_input_binary, true, 0.5);
                                                        // link input type for hidden outnodes is always range (-a,a) and p = 0.5 always

                                                return 1;
                                            }

			float							learningRate;
};

class outputLayer : public nnLayer
{
	// setup
	public:
                                            outputLayer(network_description & net, unsigned int layerIndex) : nnLayer(layerIndex)
                                            {
                                                unsigned int i;
                                                // create the list of nodes done in nnLayer
                                                nodes = new vector<outputNode*> (net.outputNodes());
                                                for (i = 0; i < net.outputNodes(); i++)
                                                    nodes->operator[](i) = new outputNode(net, i);
                                            }

    virtual									~outputLayer()
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    delete (*nodeI);

                                                delete nodes;
                                            }

    virtual void							setLinkWeights(twoDFloatArray * weightArray)	// no longer used
                                            {
                                                // no longer used
                                                throw ;
                                            }

    virtual	void							setNodeBiases(vector<float> * nodeBiasArray)
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                    (*nodeI)->setBias(nodeBiasArray->operator[]((*nodeI)->nodeIndex()));
                                                }
                                            }

				
	private:
			vector<outputNode*>			* 	nodes;
			vector<outputNode*>::iterator	nodeI;

		
	// save & retrieve
	public:
            void							storeOn(stringstream * strOut)
                                            {
                                                (*strOut) << "comment(Storing the output layer)\n";

                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                    (*nodeI)->storeOn(strOut, nnLayer::index);
                                                }

                                            }

			void							linkWeights(twoDFloatArray * weightArray);
			void							nodeBiases(vector<float> * nodeArray);	
            void							trainingError(vector<float> * errorVector)
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                    errorVector->operator[]((*nodeI)->nodeIndex()) = (*nodeI)->lastErrorValue();
                                                }
                                            }


            vector<outputNode*>			*	nodeList()
                                            {
                                                return nodes;
                                            }

	
	// training
	public:
            int								randomise(random_device & randSeed)
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    (*nodeI)->randomise(randSeed);

                                                return 1;
                                            }

            void							setDesiredValues(vector<float> * desiredVals)
                                            {
                                                if (desiredVals->size() == nodes->size())
                                                    for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                        (*nodeI)->setDesiredValue(desiredVals->operator[]((*nodeI)->nodeIndex()));
                                                else
                                                    ; // do something
                                            }


	// run
	public:
//			vector<float>			*		outputVector();
            void							returnOutputVector(vector<float> * outVec)
                                            {
                                                float val;
                                                unsigned int i;

                                                if (outVec->size() == nodes->size())
                                                    for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    {
                                                        val = (*nodeI)->value();
                                                        i = (*nodeI)->nodeIndex();
                                                        outVec->operator[](i) = val;
                                                    }
                                                else
                                                    throw;	// throw an error

                                            }


			status_t						waitForActivation()
                                            {
                                                status_t endValue;

                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                {
                                                    // reset all the CEvents so that the main thread blocks until all the link values are in
                                                    endValue = (*nodeI)->waitForActivation();
                                                    if (endValue == 0)
                                                        break;	// !!!!!!!!! do something more useful !!!!!!!!!!!!
                                                }
                                                return endValue;
                                            }

            void							blockTillValue()
                                            {
                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)	// wait till all values are in using CEvent::Lock()
                                                    (*nodeI)->blockTillValue();

                                            //	for (i = 0; i < nodes->size(); i++) // release all semaphores	(not used with CEvents)
                                            //		(nodes->operator[](i))->mainThreadRelease();
                                            }

	private:

};

#endif
