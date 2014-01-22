/*
 *
 * networkDescription.hpp All the parameters of a network. One of these objects plus a networkFile object added
 * together gives a network.
 *
 */

#ifndef _networkDescription_h
#define _networkDescription_h

#include <string>
#include <iostream>

const int maxNodes = 64;
const int maxLayers = 2;	// input layer is layer 0

enum layer_modifier { BIAS_NODE, TRANSITION_SIGMOID, TRANSITION_LINEAR, TRANSITION_BINARY };
enum node_modifier { INPUT_BINARY, INPUT_UNIFORM, INPUT_CONTRADICTORY };

struct nodeData
{
  public:
	node_modifier inputType;
	float p;
	bool pIsOneHalf;
};

struct layerData
{
  public:
	unsigned int nodeCount;
	twoDFloatArray * weights;
	vector<float> * biases;
	bool hasBiasNode;
	layer_modifier transition;
	vector<float> * nodeValues;
	vector<nodeData> * nodeInfo;
};


class network_description
{
	public:
                        network_description() { }

                        network_description(int inputNodes, int hiddenNodes, int outputNodes, float newLearningRate)
                        {
                            inputNodeCount = inputNodes;
                            hiddenNodeCount = hiddenNodes;
                            outputNodeCount = outputNodes;
                            learningRate = newLearningRate;
                            name = "network-addTopology";
                            inputLayerBiasNode = false;
                        }

                        network_description(int inputNodes, int hiddenNodes, int outputNodes, float newLearningRate, const std::string netName)
                        {
                            inputNodeCount = inputNodes;
                            hiddenNodeCount = hiddenNodes;
                            outputNodeCount = outputNodes;
                            learningRate = newLearningRate;
                            name = netName;
                            inputLayerBiasNode = false;
                        }

                        ~network_description() {}

    network_description operator=(network_description other)
                        {
                            inputNodeCount = other.standardInputNodes();
                            hiddenNodeCount = other.hiddenNodes();
                            outputNodeCount = other.outputNodes();
                            learningRate = other.learningRate;
                            momentum = other.momentum;
                            inputLayerBiasNode = other.inputLayerBiasNode;
                            name = other.name;

                            return other;
                        }

	// ======= Access =======
	unsigned int		standardInputNodes()	{ return inputNodeCount; }
	unsigned int		inputNodes()			{ return inputLayerBiasNode ? inputNodeCount + 1 : inputNodeCount; }
	unsigned int		hiddenNodes()			{ return hiddenNodeCount; }
	unsigned int		outputNodes()			{ return outputNodeCount; }
	bool				hasInputLayerBiasNode() { return inputLayerBiasNode; }
	float				trainingLearningRate()	{ return learningRate; }
	float				trainingMomentum()		{ return momentum; }
	std::string			networkName()			{ return name; }

	// ======= Assignment =======

	void				setStandardInputNodes(unsigned int noInputNodes) { inputNodeCount = noInputNodes; }
//	void				setInputNodes(unsigned int noInputNodes, bool hasInputLayerBiasNode) { inputLayerBiasNode ? inputNodeCount + 1 : inputNodeCount; }
	void				setHiddenNodes(unsigned int noHiddenNodes) { hiddenNodeCount = noHiddenNodes; }
	void				setOutputNodes(unsigned int noOutputNodes) { outputNodeCount = noOutputNodes; }
	void				setInputLayerBiasNode(bool reallyDoesHaveAnInputBiasNode) { inputLayerBiasNode = reallyDoesHaveAnInputBiasNode; }
	void				setTrainingLearningRate(float newLearningRate) { learningRate = newLearningRate; }
	void				setTrainingMomentum(float newMomentum) { momentum = newMomentum; }
    void				setNetworkName(std::string newName) { name = newName; }
    void				setMomentum(float newMomentum) { momentum = newMomentum; }


	protected:
	float			momentum;

	private:

	unsigned int 		inputNodeCount;
	unsigned int 		hiddenNodeCount;
	unsigned int 		outputNodeCount;
	vector<layerData>*	layers;
	bool				inputLayerBiasNode;
	float				learningRate;
	std::string			name;

};


#endif // _networkDescription_h
