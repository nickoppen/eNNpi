#ifndef _nnLink_h
#define _nnLink_h

#include <sstream>
#include <random>
#include "nnNodeBase.hpp"
#include "inputType.hpp"

class nnLink
{
	public:
                        nnLink(nnNode * inNode, nnNode * outNode, unsigned int newIndex)
                        {
                            theInputNode = inNode;
                            theOutputNode = outNode;
                            index = newIndex;
                            lastWeightChange = 0.0;
                        }

                        nnLink() { }

    void				setEnds(nnNode * inNode, nnNode * outNode)
                        {
                            theInputNode = inNode;
                            theOutputNode = outNode;
                        }

	
	public:
	// training
    float				outputLevel()				// request the output value (called in training to request the link's level from its input node and weight)
                        {
                            return weight * theInputNode->value();
                        }

    void				adjustWeight(float delta)	// addjust the weight in training
                        {
                            weight += delta;
                            outputErrorCalculated = false;
                        }


    int					randomise(std::random_device & randSeed, const int input_type, bool pEqualsOneHalf, float p, int faninToNode)
							// set the weight to a random number
							// pEqualsOneHalf == true assumes p == 0.5
							// for uniform inputs p is the upper most positive value expected
                        {
                            int newRand;
                            float numerator;
                            float denominator;
                            float weightMax;

                            if ((input_type == node_input_binary) && pEqualsOneHalf)
                            {
                                numerator = (float)5.1;
                                denominator = sqrt((float)faninToNode);
                            }
                            else if ((input_type == node_input_binary) && !pEqualsOneHalf)
                            {
                                numerator = (float)2.55;
                                denominator = sqrt((float)faninToNode * p * (1 - p));
                            }
                            else if ((input_type == node_input_bipolar) && pEqualsOneHalf)
                            {
                                numerator = (float)2.55;
                                denominator = sqrt((float)faninToNode);
                            }
                            else if ((input_type == node_input_bipolar) && !pEqualsOneHalf)
                            {
                                numerator = (float)1.28;
                                denominator = sqrt((float)faninToNode * p * (1 - p));
                            }
                            else if (input_type == node_input_uniform)
                            {
                                numerator = (float)4.4;
                                denominator = p * sqrt((float)faninToNode);
                            }
                            else
                            {
                                throw;	// opps!
                            }

                            weightMax = numerator / denominator;

                            newRand = randSeed();
                            weight = (weightMax - ((float)newRand / ((float)randSeed.max() / (2 * weightMax))));
                            outputErrorCalculated = false;

                            return newRand;
                        }


    float 				adjustWeightReturnOutputError(float nodeValueTimesLearningRate, float learningRate, float momentum)
                        {
                            float outputError;
                            float currentWeightChange;

                            outputError = theOutputNode->adjustBiasReturningOutputError();

                            currentWeightChange = nodeValueTimesLearningRate * outputError;
                            weight += currentWeightChange + (lastWeightChange * momentum);
                            lastWeightChange = currentWeightChange;

                            return outputError;
                        }


	// run
    void				activate(float inValue)	// call during run to push the input value to the output node
                        {
                            theOutputNode->activationFromLink(weight * inValue);
                        }


	// access
	float				getWeight() { return weight; }
	unsigned int		linkIndex() { return index; }
	
	// static class
//	public:
	
    void				storeOn(stringstream * strOut, unsigned int layerNo, unsigned int nodeNo)
                        {

 //                           (*strOut) << weight;
 //                           (*strOut) << layerNo;
 //                           (*strOut) << nodeNo;
 //                           (*strOut) << index;
                            (*strOut) << "link(";
                            (*strOut) << layerNo;
                            (*strOut) << ",";
                            (*strOut) << nodeNo;
                            (*strOut) << ",";
                            (*strOut) << index;
                            (*strOut) << ",";
                            (*strOut) << weight;
                            (*strOut) << ")\n";
                        }

    void				setWeight(float newWeight)
                        {
                            weight = newWeight;
                        }

	float				linkWeight() { return weight; }; 
	
	private:
	float				weight;
	nnNode	*			theInputNode;
	nnNode	*			theOutputNode;
	unsigned int		index;

	float				lastWeightChange;	// multiply it by the momentum as add it to the current weight change
	
	float				scaledOutputError;
	bool				outputErrorCalculated;
};

#endif
