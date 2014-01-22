#ifndef _nn_h
#define _nn_h

#define ennVersion "(1.1.0)"

#include <sys/stat.h> // POSIX only

#include <sstream>
#include <random>
#include <vector>
#include "nnFile.hpp"
//#include "dataFile.hpp"
#include "twoDFloatArray.hpp"
//#include "nnLayer.hpp"

const int maxNodes = 64;
const int maxLayers = 3;	// input layer is layer 0

enum layer_modifier { BIAS_NODE, TRANSITION_SIGMOID, TRANSITION_LINEAR, TRANSITION_BINARY };
enum node_modifier { INPUT_BINARY, INPUT_UNIFORM, INPUT_BIPOLAR };

struct nodeData
{
  public:
	node_modifier inputType;
	float p;			/// P(x=1) = p
	bool pIsOneHalf;
	float nodeValue;
	float bias;
	vector<float> * incomingWeights;
};

struct layerData
{
  public:
	unsigned int nodeCount;
	bool hasBiasNode;
	layer_modifier transition;
	vector<nodeData> * nodeInfo;
};


typedef void (*funcRunCallback)(const int, void *);
typedef void (*funcTrainCallback)(void *);
typedef void (*funcTestCallback)(const int, vector<float>*, vector<float>*, vector<float>*, vector<float>*, void *);

class nn
{
	public:
						nn(int inputLayerWidth, int hiddenLayerWidth, int outputLayerWidth, float learningRateParam, string & newName)
						/*
						 * Create a new network with
						 * inputLayerWidth input nodes,
						 * hiddenLayerWidth hidden nodes,
						 * outputLayerWidth output nodes and
						 * learningRateParam as the learning rate with
						 * newName as the network name
						 */
                        {
							vector<unsigned int> widths(3);		// only while the number of layers is restricted to 3
							widths[0] = inputLayerWidth;
							widths[1] = hiddenLayerWidth;
							widths[2] = outputLayerWidth;

							setNetworkTopology(&widths);
							networkName = newName;
							trainingLearningRate = learningRateParam;

                            majorVersion = minorVersion = revision = 0;
                            networkName = newName;

                        }

                        nn(NNFile * newFile)
                        /*
                         * Reconstruct a network from a saved file with the wrapper newFile
                         */
                        {
                        	newFile->readInFile((void*)this);
                        };

                        ~nn()
                        /*
                         * The network object destructor.
                         *
                         * ALWAYS make sure you call this function.
                         *
                         */
                        {
                        	delete errorVector;
//                            delete theInputLayer;
//                            delete theHiddenLayer;
//                            delete theOutputLayer;
                        }

						
	// operation
			void		run(NNFile * inFile, funcRunCallback runComplete = NULL)
			/*
			 * Run the contents of the data file wrapped by inFile calling the runComplete callback
			 * for each line.
			 *
			 *	typedef void (*funcRunCallback)(const int index, void * thisNetwork);
			 *	index is the index of the row that has just been run
			 *	theNetwork is a void pointer to this object.
			 *
			 *	Call ((nn*)theNetwork)->runResult(vector<float>* existingVector) to retrieve the result
			 *
			 */
                        {
							inFile->readInFile((void*)this);
                        }

            void		run(vector<float> * inputVector, funcRunCallback runComplete = NULL, const int index = 0)
            /*
             * Pass inputVector to the input layer and trigger it to execute the network logic. Call the
             * runComplete callback if it is not NULL.
             *
			 *	typedef void (*funcRunCallback)(const int index, void * thisNetwork);
			 *	index is the index of the row that has just been run
			 *	theNetwork is a void pointer to this object.
			 *
			 *	Call ((nn*)theNetwork)->runResult(vector<float>* existingVector) to retrieve the result
			 *
			 * Note: this version is not multi threaded so waitForActivation and blockTillValue do nothing
             */
                        {
            				unsigned int i;

            				for (i=0; i < (*layers)[0].nodeCount; i++)
            					(*layers)[0].nodeInfo->operator[](i).nodeValue = (*inputVector)[i];		// copy in the input

            	cout << "index: " << index << " feedforward - first call \n";
            				feedForward(1);													// calulates the node values of layer 1 from the input vector

                cout << "index: " << index << " feedforward - second call call \n";
            				feedForward(2);													// calculates the node values of layer 2, the output layer from the node values of layer 1

                            if (runComplete != NULL)
                                runComplete(index, (void*)this);
                        }

            void		run(vector<float> * inputVector, vector<float> * outputVector)
            /*
             * Run a single input vector and return the result. This call is designed to
             * run synchronously.
             */
                        {
                            run(inputVector);
                            // wait for the result
                            runResult(outputVector);
                        }

            vector<float> * runResult(vector<float> * outputVector)
			/*
			 * Set and return outputVector from the last run.
			 *
			 * Call this quickly - I'm not sure how long it will be before the result is written
			 * over by the next output.
			 *
			 */
                        {
            				unsigned int outI;

            				for (outI = 0;  outI < (*layers)[2].nodeInfo->size(); outI++)
            					(*outputVector)[outI] = (*layers)[2].nodeInfo->operator[](outI).nodeValue;	// copy the contents
                            return outputVector;
                        }

            void 		train(NNFile * trFile, funcTrainCallback trComplete = NULL)
            /*
             * Train the network using the training set in the file wrapped by trFile. Call the trComplete callback once
             * when training is complete.
             *
             * typedef void (*funcTrainCallback)(void * nnObj); passes an anomymous pointer to this object back via the callback
             *
             * call ((nn*)nnObj)->trainingError(vector<float>* existingVector); to retrieve the most recent training error vector
             *
             */
                        {
                            // call train with each vector
//                            unsigned int i;
//
//                            for (i=0; i < trFile->inputLines(); i++)
//                                train(trFile->inputSet(i), trFile->outputSet(i));	// don't pass the call back because we only want it called at the end not after each training set

            				trFile->readInFile((void*)this, true);

            				// block til complete
                            incrementRevision();

                            if (trComplete != NULL)
                                trComplete((void*)this);
                        }

            void		train(vector<float> * inputVector, vector<float> * desiredVector, funcTrainCallback trComplete = NULL)
            /*
             * Train the network with the single pair, inputVector and desiredVector. Call the trComplete callback if it is not NULL
             * when training is complete.
             *
             * typedef void (*funcTrainCallback)(void * nnObj); passes an anomymous pointer to this object back via the callback
             *
             * call ((nn*)nnObj)->trainingError(vector<float>* existingVector); to retrieve the most recent training error vector
             *
             */
                        {
                            try
                            {
                                // run
                                run(inputVector);

//                               theOutputLayer->setDesiredValues(desiredVector);

//                               theHiddenLayer->waitForTraining();
//                               theHiddenLayer->train();
//                                theHiddenLayer->blockTillTrained();

//                                theInputLayer->waitForTraining();
//                                theInputLayer->train();
//                                theInputLayer->blockTillTrained();

                                if (trComplete != NULL)
                                    trComplete((void*)this);
                            }
                            catch (internal_Error & iErr)
                            {
                                cout << iErr.mesg;// << " last error:" << iErr.lastError;
                            }

                            hasChanged = true;
                        }

            status_t	trainingError(vector<float> * errorVector)
            /*
             * Return the most recent error vector generated by the most recent training set.
             *
             * Note: the errorVector must exist and be the right size
             *
             */
						{
            				if (errorVector->size() != (*layers)[2].nodeCount)
            					return FAILURE;
            				else
            		//			theOutputLayer->trainingError(errorVector);
            					;
							return SUCCESS;
						}

            void		test(NNFile * testFile, funcTestCallback testComplete = NULL)
            /*
             * Run the data component of the training file inside the wrapper testFile and
             * compare the output generated by the network to the desired output. Calculate the difference.
             *
             * The call back function funcTestCallback is called once for every line in the input file and has the following form:
             *
             * typedef void (*funcTestCallback)(const int index, vector<float>* inputVector, vector<float>* desiredOutput, vector<float>* outputVector, vector<float>* errorVector, void * thisObject);
             * index: the row number in the file
             * inputVector: a pointer to the test data vector
             * desiredOutput: a pointer to the desired output vector
             * outputVector: a pointer to the actual output from the net
             * errorVector: a pointer to a vector containing the desired minus the actual output
             * thisObject: an anonymous pointer to this object
             *
             */
						{

            				testFile->readInFile((void*)this, false);	// false to indicate that the file is NOT a training file

						}

            void		test(const int index, vector<float> * inputVector, vector<float> * desiredOutput, funcTestCallback testComplete = NULL)
            /*
             * Test a single input vector and compare the result with the givine output vector. Then
             * compare the output generated by the network to the desired output. Calculate the difference.
             *
             * The call back function funcTestCallback is called once and has the following form:
             *
             * typedef void (*funcTestCallback)(const int index, vector<float>* inputVector, vector<float>* desiredOutput, vector<float>* outputVector, vector<float>* errorVector, void * thisObject);
             * index: the row number in the file
             * inputVector: a pointer to the test data vector
             * desiredOutput: a pointer to the desired output vector
             * outputVector: a pointer to the actual output from the net
             * errorVector: a pointer to a vector containing the desired minus the actual output
             * thisObject: an anonymous pointer to this object
             *
             */
						{
            				size_t i;
            				vector<float> outputVec((*layers)[2].nodeCount);

            				// run
							run(inputVector);

							// block til value

							// compare
//							theOutputLayer->returnOutputVector(&outputVec);

							for (i = 0; i != errorVector->size(); i++)
								(*errorVector)[i] = outputVec[i] - (*desiredOutput)[i];

							if (testComplete != NULL)
								testComplete(index, inputVector, desiredOutput, &outputVec, errorVector, (void*)this);
						}

            void		randomise()
            /*
             * Randomise the weights and biases in the network thereby restarting the training cycle from a different place.
             */
						{
							random_device rd;
							unsigned int layer;
							float linkWeightVectorLength;
							unsigned int faninToNode;
							unsigned int nodeI;
							unsigned int linkI;

							cout << "TBC: randomise();\n";

							for (layer = 1; layer < 3; layer++)
							{
/*
 *                from nnLayer
 *                                                                for (nodeI = nodes->begin(); nodeI != nodes->end(); nodeI++)
                                                    (*nodeI)->randomise(randSeed, node_input_binary, true, 0.5);
                                                        // link input type for hidden outnodes is always range (-a,a) and p = 0.5 always
 *
 *
 *                from nnNodeVirtual: randomise(random_device & randSeed, const int input_type = node_input_binary, bool pEqualsOneHalf = true, float p = 0.5)
                                             for(i = 0; i < linkCount; i++)
                                            {
                                                pLink = inLinks->operator[](i);
                                                pLink->randomise(randSeed, input_type, pEqualsOneHalf, p, linkCount);
                                            }

                                            //calculate the ||w|| (the "length" of the weight vectors from the links)
                                            for(i = 0; i < inLinks->size(); i++)
                                                linkWeightVectorLength += pow((inLinks->operator[](i))->getWeight(), 2);
                                            linkWeightVectorLength = sqrt(linkWeightVectorLength);

                                            bias = (linkWeightVectorLength - ((float)randSeed() / ((float)randSeed.max() / (2 * linkWeightVectorLength))));
 *
 *			from nnLink
 *
 *										// set the weight to a random number
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
 *
 */
							}

							hasChanged = true;
							incrementMinorVersion();
						}

	// access

			status_t 	saveTo(string * strPath)
			/*
			 * Save the network to a file called <network Name>_<majorVersion>_<minorVersion>_<revision>.enn in the path supplied in string object strPath.
			 *
			 * Note: if you have the path name already as a C string call saveTo(const char *) rather than this function
			 *
			 */
			{
				return saveTo(strPath->c_str());
			}

			status_t	saveTo(const char * cstrPath)
			/*
			 * Save the network to a file called <network Name>_<majorVersion>_<minorVersion>_<revision>.enn in the path supplied in C string cstrPath
			 */
			{
				fstream * pFile;
				status_t rVal;
				char cstrPathFile[255];	// dumb
				char cstrFileName[25]; // dumb

				if (checkExists(cstrPath, false))
				{
					sprintf(cstrPathFile, "%s//%s", cstrPath, defaultName(cstrFileName));

					pFile = new fstream();
					pFile->open(cstrPathFile, ios::out);
					rVal = saveTo(pFile);
					pFile->close();
					delete pFile;

					return rVal;
				}
				else
					throw format_Error(ENN_ERR_NON_FILE);

				return SUCCESS;
			}

			status_t	saveTo(fstream * pFile)
			/*
			 * Save the network to the file stream pointed to by pFile. The name of the file will not be changed.
			 */
			{
				string strContent;
				status_t rVal;
				rVal = saveOn(&strContent);
				(*pFile) << strContent;

				return rVal;
			}

			// save to disk
			status_t	saveOn(string * strOut)
			/*
			 *  save the net in the given existing string
			 */
			{
				stringstream ss;

				ss.precision(8);

				ss << "version" << ennVersion << "\nname(" << networkName << "," << majorVersion << "," << minorVersion << "," << revision << ")\n" ;

				ss << "networkTopology(" << (*layers)[0].nodeCount << "," <<  (*layers)[1].nodeCount <<  "," <<  (*layers)[2].nodeCount << ")\n";

				ss << "learning(" << trainingLearningRate << "," << trainingMomentum << ")\n";

				// call the detail storage process here
//				theInputLayer->storeOn(&ss);
//				theHiddenLayer->storeOn(&ss);
//				theOutputLayer->storeOn(&ss);

				hasChanged = false;

				(*strOut) = ss.str();

				return SUCCESS;
			}

	// Modify
			status_t	alter(int newIn, int newHidden, int newOut)
			/*
			 * Alter the topology of the network to be
			 * newIn: the new number of input nodes
			 * newHidden: the new number of hidden nodes
			 * newOut: the new number of output nodes
			 *
			 * This will randomise the network and increment the major version resetting the minorVerions and revision
			 *
			 */
			{
//				unsigned int layerNo = 0;

//				delete theInputLayer;
//				delete theHiddenLayer;
//				delete theOutputLayer;

//                net.setHiddenNodes(newHidden);
//                net.setOutputNodes(newOut);
//                net.setStandardInputNodes(newIn);

//                theInputLayer = new inputLayer(net, layerNo++);		// deleted in ~nn
//                theHiddenLayer = new hiddenLayer(net, layerNo++);	// deleted in ~nn
//                theOutputLayer = new outputLayer(net, layerNo++);	// deleted in ~nn

//                theInputLayer->connectNodes(theHiddenLayer->nodeList());
//                theHiddenLayer->connectNodes(theOutputLayer->nodeList());

                randomise();
                incrementMajorVersion();

                hasChanged = true;
				return SUCCESS;
			}

//			status_t	alter(unsigned int layer, layer_modifier mod, bool boolAdd = true)
//			/*
//			 * Alter a layer within the network. Currently you can only add or remove a bias node from layer zero (the input layer)
//			 *
//			 * This will randomise the network and increment the major version resetting the minorVerions and revision
//			 *
//			 */
//			{
//				network_description newNet;
//
//				newNet = net;	// keep all the old values
//
////				delete theInputLayer;
////				delete theHiddenLayer;
////				delete theOutputLayer;
//
//                newNet.setInputLayerBiasNode(boolAdd);
//
//                setup(newNet);
//
//                randomise();
//                incrementMajorVersion();
//
//                hasChanged = true;
//
//                return SUCCESS;
//			}

            char *	defaultName(char * buffer)
            /*
             * Return the default name for the network, which is: <network Name>_<majorVersion>_<minorVersion>_<revision>.enn
             *
             * Note: the calling function must make sure that there is enough room in the buffer
             *
             */
            {
            	sprintf(buffer, "%s_%d_%d_%d.enn", networkName.c_str(), majorVersion, minorVersion, revision);

            	return buffer;
            }

			bool		needsSaving() { return hasChanged; }	// Return true if the network has changed since it was last saved.

	// Build - callbacks for the networkFile that is reading in the netowrk from a .enn file

			void setNetworkTopology(vector<unsigned int> * layerWidths)
			{

				cout << "Layer widths:" << layerWidths->size() << " - " << (*layerWidths)[0] << " " << (*layerWidths)[1] << " " << (*layerWidths)[2] << "\n";

				layers = new vector<layerData>(layerWidths->size());
//				layers = new vector<layerData>(3);

				setupLayer(&((*layers)[0]), (*layerWidths)[0], 0); 	// arg 2 is the previous layer width therefore 0 for the input layer

				setupLayer(&((*layers)[1]), (*layerWidths)[1], (*layerWidths)[0]);

				setupLayer(&((*layers)[2]), (*layerWidths)[2], (*layerWidths)[1]);
			}

			void setNodeBias(unsigned int layer, unsigned int node, float bias)
			{
				cout << "bias layer " << layer << " node " << node  << " bias " << bias << "\n";
				layers->operator[](layer).nodeInfo->operator[](node).bias = bias;
			}

			void setLinkWeight(unsigned int layer, unsigned int fromNode, unsigned int toNode, float weight)
			{
				cout << "weight layer: " << layer << " from " << fromNode << " to " << toNode << "weight " << weight << "\n";
				layers->operator[](layer).nodeInfo->operator[](toNode).incomingWeights->operator[](fromNode) = weight;
			}

			void setName(string * name)
			{
				cout << "name:" << (*name) << "\n";
				networkName = *name;
			}

			void setVersion(unsigned int major, unsigned int minor, unsigned int revis)
			{
				cout << "version " << major << " " << minor << " " << revis << "\n";
				majorVersion = major;
				minorVersion = minor;
				revision = revis;
			}

			void setTrainingLearningRate(float learningRate)
			{
				cout << "setting LR:" << learningRate << "\n";
				trainingLearningRate = learningRate;
			}

			void setTrainingMomentum(float momentum)
			{
				cout << "setting momentum:" << momentum << "\n";
				trainingMomentum = momentum;
			}

			void setHasBiasNode(unsigned int layer, bool hasBiasNode)
			{
				cout << "layer: " << layer << "bias node:" << hasBiasNode << "\n";
				(*layers)[layer].hasBiasNode = hasBiasNode;		// the node is added at setup time and is only used if hasBiasNode is true
			}

	// Access

			unsigned int layerZeroWidth()
			{
				return (*layers)[0].nodeCount;
			}

			unsigned int layerNWidth()
			{
				return (*layers)[2].nodeCount;
			}

    // Setup
	private:
            void		incrementRevision() { revision++; }
            void		incrementMinorVersion() { minorVersion++; revision = 0; }
            void		incrementMajorVersion() { majorVersion++; minorVersion = revision = 0; }

            void		setupLayer(layerData * layer, unsigned int width, unsigned int previousLayerWidth)
            {
            	unsigned int nodeI;

            	cout << "layer width: " << width << " prev:" << previousLayerWidth << "\n";

            	layer->nodeCount = width;
            	layer->nodeInfo = new vector<nodeData>(width + 1); // add the space now for the bias node
            	layer->transition = TRANSITION_SIGMOID;
            	layer->hasBiasNode = false;

            	for(nodeI = 0; nodeI < width; nodeI++)
            	{
            		cout << "adding node: " << nodeI << " content\n";
            		layer->nodeInfo->operator[](nodeI).inputType = INPUT_UNIFORM;
            		layer->nodeInfo->operator[](nodeI).p = 0.5;
            		layer->nodeInfo->operator[](nodeI).pIsOneHalf = true;

            		if (previousLayerWidth != 0)	// previous  == 0 indicates that the layer is the input layer therefore does not need any incoming weights
            		{
            			layer->nodeInfo->operator[](nodeI).incomingWeights = new vector<float>(previousLayerWidth + 1); // add one in case the previous layer has a bias node

            			cout << "space for incoming weights: " << layer->nodeInfo->operator[](nodeI).incomingWeights->size() << "\n";
            		}
            		else
            			cout << "no incoming weight space allocated\n";

            		//layer->nodeInfo->operator[](nodeI).bias = rand();
            	}

            	// and set up the bias node in case it gets switched on later
        		cout << "adding bias node content\n";
        		layer->nodeInfo->operator[](nodeI).inputType = INPUT_BINARY;
        		layer->nodeInfo->operator[](nodeI).p = 1;
        		layer->nodeInfo->operator[](nodeI).pIsOneHalf = false;
        		layer->nodeInfo->operator[](nodeI).nodeValue = 1;

            }

            void		setup()//network_description newNet, bool createWeightArrays = false)
            {

                errorVector = new vector<float>((*layers)[2].nodeCount);	// deleted in the destructor

//                if (createWeightArrays)
//                	randomise();
            }

    // Other
            bool checkExists(const char * fileName, bool boolShouldBeFile = true)
            {
            	struct stat fileAtt;

            	if (stat(fileName, &fileAtt) != 0)
            		return false;
            	else
            		if (boolShouldBeFile)
            			return S_ISREG(fileAtt.st_mode);
            		else
            			return S_ISDIR(fileAtt.st_mode);

            }

            void feedForward(unsigned int targetLayer)
            {
            	unsigned int inputI;	// vector iterator: iterates over the input vector
            	unsigned int outputI;	// iterates over the target vector
            	unsigned int prevLayer = targetLayer -1;
            	unsigned int incomingNodeCount;	// one greater if prev layer has a bias node
            	float sum = 0;

//            	cout << "output vecotr size:" << nodes->size() << "\n";
            	for (outputI = 0; outputI < (*layers)[targetLayer].nodeCount; outputI++)
            	{
            		incomingNodeCount = ((*layers)[prevLayer].hasBiasNode ? (*layers)[prevLayer].nodeCount + 1 : (*layers)[prevLayer].nodeCount);
//            		cout << "output node:" << outputI << " Input vector size:" << inputData->size() << "\n";
            		for (inputI = 0; inputI < incomingNodeCount; inputI++)
            		{
            			sum += ((*layers)[prevLayer].nodeInfo->operator[](inputI).nodeValue) * (*layers)[targetLayer].nodeInfo->operator[](outputI).incomingWeights->operator[](inputI);
            			cout << "target node:" << outputI << " input node:" << inputI << " sum :" << sum << "\n";
            		}
            		switch ((*layers)[targetLayer].transition)
            		{
            		case TRANSITION_LINEAR:
//            			break;
            		case TRANSITION_BINARY:
            			cout << "Only SIGMOID transitions have been implemented.\n";
//            			break;
            		case TRANSITION_SIGMOID:
            		default:
            			(*layers)[targetLayer].nodeInfo->operator[](outputI).nodeValue = 1 / (1 + exp((double)(-1 * (sum + (*layers)[targetLayer].nodeInfo->operator[](outputI).bias))));		// sigmoid of the sum plus the bias
            			break;
            		}
            		cout << "node: " << outputI << " value " << (*layers)[targetLayer].nodeInfo->operator[](outputI).nodeValue << "\n";
            		// from outNode:: (1 / (1 + exp((double)(-1 * biasPlusActivationQuant))))
            	}
            	cout << "end feedforward \n\n";

            }
			
	private:
    vector<layerData>  * layers;
    float				 trainingLearningRate;
    float				 trainingMomentum;

	// house keeping
	bool				hasChanged;					// set to true after randomisaton or training

	// testing
	vector<float>	*	errorVector;				// pass a pointer to this vector in the test callback

	// identificaton
	unsigned int		majorVersion;
	unsigned int		minorVersion;
	unsigned int		revision;					// a number to distinguish different versions of the same net
    string				networkName;
};

#endif
