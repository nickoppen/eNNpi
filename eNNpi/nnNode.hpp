#ifndef _nnNode_h
#define _nnNode_h

#include <sstream>
#include <float.h>

#include "nnLink.hpp"
#include "networkDescription.hpp"
#include "nnNodeVirtual.hpp"		// includes the virtual base classes for actual nodes

// C decls
//unsigned int hidden_runner(void * arg)     // used in hiddenNode to start the next layer of pushes/
//{
//    ((hiddenNode*)arg)->pushValue();
//    return 1;
//}



class outputNode :  public outNode
{
	public:
                            outputNode(network_description & net, unsigned int index) : outNode(net, index)
                            {
                                nodeName = "Output Node";
                                setInLinks(new vector<nnLink*>(net.hiddenNodes()));

                                //syncEventHandle = CreateEvent(NULL, TRUE, FALSE, NULL);
                                // the QT semaphore is already instantiated
                            }

    virtual					~outputNode()
                            {
                                //CloseHandle(syncEventHandle);
                                // the semaphore is an instance variable and will be deleted automatically
                            }

	void					setWeights(vector<float> *);

	// run
    bool					activationFromLink(float value)
                            {
                                if (outNode::activationFromLink(value))
                                {
                                    //SetEvent(syncEventHandle);-----------------------------------------------------------------<<<
                                    return true;
                                }
                                else
                                    return false;
                            }

    status_t   				waitForActivation()
                            {
                                // use a Events to co-ordinate the run
                                //return ResetEvent(syncEventHandle);
                                //sem.acquire();
    							return SUCCESS;
                            }

    void					blockTillValue()
                            {
                                //WaitForSingleObject(syncEventHandle, INFINITE);
                                //sem.acquire();
                            }

    void					mainThreadRelease()
                            {
                                //	release_sem(currentSemaphoreId);
                                //	ReleaseSemaphore(currentSemaphoreId, 1, NULL);
                                //	syncEvent->SetEvent();
                            }

	
	// output
    void					storeOn(stringstream * strOut, unsigned int layerNo)
                            {
                                outNode::storeOn(strOut, layerNo);
                            }
	
	// training
    void					setDesiredValue(float val)
                            {
                                // assumes that the input pattern has been run before then desired output is set.
                                desiredValue = val;
                                outputError = nodeValue * (1 - nodeValue) * (desiredValue - nodeValue);

                            }

    float					adjustBiasReturningOutputError()
                            {
                                float currentBiasChange;

                                currentBiasChange = outputError * learningRate;
                                bias += currentBiasChange + (lastBiasChange * momentum);
                                lastBiasChange = currentBiasChange;

                                return outputError;
                            }

    float					lastErrorValue() { return outputError; };
	
	private:
	// run

	// train
	float					desiredValue;
	float					outputError; 		// set when the desired value is set

};

class hiddenNode :  public inNode, virtual public outNode
{
	// setup
	public:
                            hiddenNode(network_description & net, unsigned int newIndex) : inNode(net, newIndex), outNode(net, newIndex)
                            {
                                nodeName = "Hidden Node";
                                setInLinks(new vector<nnLink*>(net.inputNodes()));
                                setOutLinks(new vector<nnLink*>(net.outputNodes()));

                                hiddenError = 0;

//                                syncEventHandle = CreateEvent(NULL, true, false, NULL);
                            }

    virtual					~hiddenNode()
                            {
                                unsigned int i;

                                for (i = 0; i < (outNode::getInLinks())->size(); i++)
                                    delete outNode::inLink(i);

                                for (i = 0; i < (inNode::getOutLinks())->size(); i++)
                                    delete inNode::outLink(i);

                            //	delete syncEvent;
                            //    CloseHandle(syncEventHandle);
                            }

    void					connectTo(vector<outputNode*> * outNodes)	// create nnLinks for all the nodes in the given lists and keep the links in inLinks and outLinks
                            {
                                // connect the node to the input and output nodes
                                outputNode * otherNode;
                                unsigned int linkIndex = 0;
                                vector<outputNode*>::iterator vit;

                                for(vit = outNodes->begin(); vit != outNodes->end(); vit++)
                                {
                                    otherNode = *vit;
                                    otherNode->addInLink(inNode::connectTo((outNode*)otherNode, linkIndex++));			// add the new link to the hidden node's list
                                }
                            }

//	void					addLink(nnLink * link);						// links to input and output nodes are created by hidden nodes (see connectTo())

    void					storeOn(stringstream * strOut, unsigned int layerNo)
                            {
                                outNode::storeOn(strOut, layerNo);
                                inNode::storeOn(strOut, layerNo);
                            }

	// training
	public:
//	void					randomise();	// sets its value and the value of its input links and output links to a random number
    int						train()		// assumes that input layers and output layers have been loaded with values to train on
                            {
                                float sumOfLinkErrors;
                                float nodeValueTimesLearningRate;
                                float currentBiasChange;

                                nodeValueTimesLearningRate = learningRate * nodeValue;
                                sumOfLinkErrors = 0;
                                for (outLinkI = outLinks->begin(); outLinkI != outLinks->end(); outLinkI++)
                                {
                                    sumOfLinkErrors += ((*outLinkI)->linkWeight() * (*outLinkI)->adjustWeightReturnOutputError(nodeValueTimesLearningRate, learningRate, momentum));
                                }	// Rao,Rao page 126 above and below

                                hiddenError = nodeValue * (1 - nodeValue) * sumOfLinkErrors;

                                currentBiasChange = hiddenError * learningRate;	// Rao,Rao page 127
                                bias += currentBiasChange + (lastBiasChange * momentum);
                                lastBiasChange = currentBiasChange;

                                // training complete so release the main thread
                                //return SetEvent(syncEventHandle);
                                return 1;

                            }

	float					adjustBiasReturningOutputError() { return hiddenError; }; // the bias has already been adjusted

    status_t				waitForTraining()	// may be able to share semaphore code with output node
                            {
                                //return ResetEvent(syncEventHandle);
    							return SUCCESS;
                            }

    void					blockTillTrained()
                            {
//                                DWORD rVal;
//                                rVal = WaitForSingleObject(syncEventHandle, INFINITE);
//                                if (rVal == WAIT_FAILED)
//                                    throw internal_Error(ENN_ERR_TRAIN_WAITFOREVENT);

                            }

    void					mainThreadRelease()
                            {
                            //	release_sem(currentSemaphoreId);	no need with CEvents
                            }


	private:
	float					hiddenError;

	// class
	public:
	static	void			setLRP(float newLRP);

	private:
	void					errForAllLinks();		// calculate the error vector by pushing the current activation to the output links and getting the error value back
	float					nodeValueError();
	void					adjustInputLinks();
	void					adjustBiasValue(float delta);
	void					pullActivation();		// get the nodes activation level by requesting each links activation level

	// run
	public:
    bool					activationFromLink(float activationLevel)
                            {
                                if (outNode::activationFromLink(activationLevel))
                                {
                            #ifdef MULTI_THREADED
                                    // multi threaded version start a new thread for each hidden node
                                    AfxBeginThread(hidden_runner, (LPVOID)this, THREAD_PRIORITY_BELOW_NORMAL);
                            #else
                                    // single threading, the last thread in continues with the next layer
                                    inNode::pushValue();	// call pushValue to pass the current node value to all links
                            #endif
                                    return true;
                                }
                                else
                                    return false;
                            }

	

};

class inputNode :  public inNode
{
	public:
                            inputNode(network_description & net, unsigned int index) : inNode(net, index)			// calls the super class constructor
                            {
                                nodeName = "Input Node";
                                setOutLinks(new vector<nnLink*>(net.hiddenNodes()));

                                //syncEventHandle = CreateEvent(NULL, TRUE, FALSE, NULL);
                            }

    virtual					~inputNode()
                            {
                                //	delete syncEvent;
                                //    CloseHandle(syncEventHandle);
                            }

    void					value(float newValue)	// sets the nodes value
                            {
                                nnNode::value(newValue);
                            }

    void					connectTo(vector<hiddenNode*> * hiddenNodes)	// create nnLinks for all the nodes in the given lists and keep the links in inLinks and outLinks
                            {
                                vector<hiddenNode*>::iterator vit;
                                hiddenNode * otherNode;
                                unsigned int linkNo = 0;

                                for(vit = hiddenNodes->begin(); vit != hiddenNodes->end(); vit++)
                                {
                                    otherNode = *vit;
                                    otherNode->addInLink(inNode::connectTo((outNode*)otherNode, linkNo++));
                                }
                            }

    void					storeOn(stringstream * strOut, unsigned int layerNo)
                            {
//    								if (layerNo == 0)
//    									(*strOut) << "comment(inputNode: " << nnNode::index << ")\n";
                                inNode::storeOn(strOut, layerNo);
                            }


	// train
    int						train()
                            {
                                float nodeValueTimesLearningRate;

                                nodeValueTimesLearningRate = learningRate * nodeValue;

                                for (outLinkI = outLinks->begin(); outLinkI != outLinks->end(); outLinkI++)
                                {
                                    (*outLinkI)->adjustWeightReturnOutputError(nodeValueTimesLearningRate, learningRate, momentum);
                                }	// we don't need to do anything with the returned value because intput nodes don't have a bias that needs to be adjusted

                                //return SetEvent(syncEventHandle);
                                return 1;
                            }

    status_t				waitForTraining()
                            {
                                //return ResetEvent(syncEventHandle);
    							return SUCCESS;
                            }

    void					blockTillTrained()
                            {
//                                DWORD rVal;
//                                rVal = WaitForSingleObject(syncEventHandle, INFINITE);
//                                if (rVal == WAIT_FAILED)
//                                    throw internal_Error(ENN_ERR_TRAIN_WAITFOREVENT);

                            }


	float					adjustBiasReturningOutputError() { return 0; };
};


#endif
