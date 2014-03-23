#ifndef _nn_h
#define _nn_h

#define ennVersion "(1.1.0)"

#include <sys/stat.h> // POSIX only

#include <sstream>
//#include <random>
#include <vector>
//#include <tgmath.h>
#include <math.h>
#include <stdlib.h> // rand
#include <time.h>
#include "nnFile.hpp"
#include "twoDFloatArray.hpp"

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

							setup();
							setNetworkTopology(&widths);
							resetVersion();				// randomise increments the minor version so reset it to zero
							networkName = newName;
							trainingLearningRate = learningRateParam;
                            networkName = newName;

                        }

                        nn(NNFile * newFile)
                        /*
                         * Reconstruct a network from a saved file with the wrapper newFile
                         */
                        {
                        	setup();
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
                        	errorVector->clear();
                        	cout<<"cleared err vec\n";
                        	delete errorVector;
                        	cout << "about to delete the layers\n";
                        	deleteLayers();
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
							runCallback = runComplete;
							inFile->readInFile((void*)this);
                        }

			void		run(vector<float> * inputVector, funcRunCallback runComplete, const int index = 0)
			/*
			 * a method of using run without reading the data in from a datafile
			 */
						{
							if (inputVector->size() == layerZeroWidth())
							{
								runCallback = runComplete;
								run(inputVector, index);
							}
							else
								;//throw;
						}

            void		run(vector<float> * inputVector, const int index = 0)
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

//            	cout << "index: " << index << " feedforward - first call \n";
            				feedForward(1);													// calulates the node values of layer 1 from the input vector

//                cout << "index: " << index << " feedforward - second call call \n";
            				feedForward(2);													// calculates the node values of layer 2, the output layer from the node values of layer 1

                            if (runCallback != NULL)
                                runCallback(index, (void*)this);
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
            				trainCallback = trComplete;
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

                                // calculate error vector
                                calculateError(desiredVector);

                                // train

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
            				if (testFile->fileType() == TRAIN_TEST)
            				{
           						testCallback = testComplete;
            					testFile->readInFile((void*)this, false);	// false to indicate that the file is NOT a training file
            				}
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
            				vector<float> outputVec((*layers)[2].nodeCount);
            				unsigned int i;

            				try
            				{
								// run
								run(inputVector);

								// block til value

								// compare
								calculateError(desiredOutput);
            				}
                            catch (internal_Error & iErr)
                            {
                                cout << iErr.mesg;// << " last error:" << iErr.lastError;
                            }

                            if (testComplete != NULL)		// use the recent one if it is set
                            	testCallback = testComplete;

							if (testCallback != NULL)
							{
								for (i=0; i < outputVec.size(); i++)
									outputVec[i] = (*layers)[2].nodeInfo->operator[](i).nodeValue;
								testCallback(index, inputVector, desiredOutput, &outputVec, errorVector, (void*)this);
							}
						}

            void		randomise()
            /*
             * Randomise the weights and biases in the network thereby restarting the training cycle from a different place.
             */
						{
//							random_device rd;    // random_device not supported by the cross compiler
							unsigned int layerI;
							float linkWeightVectorLength;
							unsigned int faninToNode;
							unsigned int nodeI;
							unsigned int linkI;
							nodeData * curNode;
							nodeData * remoteNode;
							float numerator = 0;
							float denominator = 0;
							float weightMax;
							float newRand;

							srand (rand_seed);

							for (layerI = 1; layerI < 3; layerI++)
							{
								for (nodeI = 0; nodeI < (*layers)[layerI].nodeCount; nodeI++)
								{
									linkWeightVectorLength = 0;
									curNode = &((*layers)[layerI].nodeInfo->operator[](nodeI));
									for (linkI=0; linkI <= curNode->incomingWeights->size(); linkI++)	// don't forget that there is an extra link in case the layer above has a unary input node
									{
										faninToNode = curNode->incomingWeights->size();
										remoteNode = &((*layers)[layerI-1].nodeInfo->operator[](linkI));
										switch (remoteNode->inputType)	// the num and demon is based on the type of the node in the layer above
										{
										case INPUT_BINARY:
//											cout << layerI << ": sending binary node index:" << linkI << " pisHalf:" << remoteNode->pIsOneHalf << " p:" << remoteNode->p << "\n";
											if (remoteNode->pIsOneHalf)
											{
				                                numerator = (float)5.1;
				                                denominator = sqrt(faninToNode);
											}
											else
				                            {
				                                numerator = (float)2.55;
				                                denominator = sqrt((float)faninToNode * curNode->p * (1 - curNode->p));
				                            }
											break;

										case INPUT_UNIFORM:
	//										cout << layerI << ": sending uniform node index:" << linkI << " pisHalf:" << remoteNode->pIsOneHalf << " p:" << remoteNode->p << "\n";
											if (remoteNode->pIsOneHalf)
											{
												numerator = (float)2.55;
												denominator = sqrt((float)faninToNode);
											}
											else
											{
												numerator = (float)1.28;
												denominator = sqrt((float)faninToNode * curNode->p * (1 - curNode->p));
											}
											break;

										case INPUT_BIPOLAR:
//											cout << layerI << ": sending bipolar node index:" << linkI << " pisHalf:" << remoteNode->pIsOneHalf << " p:" << remoteNode->p << "\n";
											if (remoteNode->pIsOneHalf)
											{
												numerator = (float)2.55;
												denominator = sqrt((float)faninToNode);
											}
											else
											{
												numerator = (float)1.28;
												denominator = sqrt((float)faninToNode * curNode->p * (1 - curNode->p));
											}
											break;
										}

										weightMax = numerator / denominator;
										//newRand = randSeed();
										newRand = rand() / weightMax;
										newRand = weightMax - ((float)rand() / (RAND_MAX / ((int)weightMax * 2)));
//										curNode->wight = (weightMax - ((float)newRand / ((float)randSeed.max() / (2 * weightMax))));
										curNode->incomingWeights->operator[](linkI) = newRand;		// not the whole story
//										cout << linkI << ": " << newRand << " ";
//										outputErrorCalculated = false;
										linkWeightVectorLength += pow(newRand, 2);
									}

									//calculate the ||w|| (the "length" of the weight vectors from the links)
									linkWeightVectorLength = sqrt(linkWeightVectorLength);
									curNode->bias = (linkWeightVectorLength - ((rand_seed = (float)rand()) / (RAND_MAX / (2 * linkWeightVectorLength))));
			//						cout << "\n";
								}
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
				unsigned int layerI;
				unsigned int nodeI;
				unsigned int linkI;
				nodeData * curNode;
				string nodeInput;
				string strTrans;
//				time_t ttime;

				ss.precision(8);

				ss << "version" << ennVersion << "\nname(" << networkName << "," << majorVersion << "," << minorVersion << "," << revision << ")\n" ;
				ss << "networkTopology(" << (*layers)[0].nodeCount << "," <<  (*layers)[1].nodeCount <<  "," <<  (*layers)[2].nodeCount << ")\n";
				ss << "learning(" << trainingLearningRate << "," << trainingMomentum << ")\n";
//				time(&ttime);
//				ss << "comment(Output date: " << asctime(localtime(&ttime)) << ")\n";	// asctime and ctime add a \n

				for (layerI = 0; layerI < 3 /*layers->size()*/; layerI++)
				{
					switch ((*layers)[layerI].transition)
					{
					case TRANSITION_SIGMOID:
						strTrans = "SIGMOID";
						break;
					case TRANSITION_LINEAR:
						strTrans = "LINEAR";
						break;
					case TRANSITION_BINARY:
						strTrans = "BINARY";
						break;
					case BIAS_NODE:
						;
						break;
					}

					ss << "layerModifier("<< layerI << ",biasNode:" << ((*layers)[layerI].hasBiasNode ? "true" : "false") << ",transition:" << strTrans << ")\n";

					for (nodeI = 0; nodeI < (*layers)[layerI].nodeCount; nodeI++)
					{
						curNode = &((*layers)[layerI].nodeInfo->operator[](nodeI));
						switch (curNode->inputType)
						{
						case INPUT_BINARY:
							nodeInput = "BINARY";
							break;
						case INPUT_BIPOLAR:
							nodeInput = "BIPOLAR";
							break;
						case INPUT_UNIFORM:
							nodeInput = "UNIFORM";
							break;
						}
						ss << "nodeModifier(" << layerI << "," << nodeI << ",p:" << curNode->p << ",pIsOneHalf:" << (curNode->pIsOneHalf ? "true" : "false") << ",input:" << nodeInput << ")\n";
						if (layerI > 0)
						{
							ss << "node(" << layerI << "," << nodeI << "," << curNode->bias << ")\n";
							for (linkI = 0; linkI < curNode->incomingWeights->size(); linkI++)
								ss << "link(" << layerI << "," << nodeI << "," << linkI << "," << curNode->incomingWeights->operator [](linkI) << ")\n";

							// output the link from the unary input node on the layer above.
							if ((*layers)[layerI-1].hasBiasNode)
								ss << "comment(The link below is from the unary input node on the layer above - which is in use.)\n";
							else
								ss << "comment(The link below is from the unary input node on the layer above - which is not in use.)\n";
							ss << "link(" << layerI << "," << nodeI << "," << linkI << "," << curNode->incomingWeights->operator [](linkI) << ")\n";

						}
					}

				}

				hasChanged = false;

				(*strOut) = ss.str();

				return SUCCESS;
			}

	// Modify
			status_t	alter(vector<unsigned int> * widths)
			/*
			 * Alter the topology of the network to be
			 *
			 * This will randomise the network and increment the major version resetting the minorVerions and revision
			 *
			 */
			{
				unsigned int layerI;
//				for (layerI=0;layerI<widths->size();layerI++)
//					cout << " widths:" << widths->operator [](layerI);
//				cout << "\n";

				if (widths->size() == 3)	// for now
				{
					for (layerI = 0; layerI < layers->size(); layerI++)
						if ((*layers)[layerI].nodeCount != (*widths)[layerI])
						{
							cout << "deleting layer:" << layerI << "\n";
							deleteLayer(layerI);	// only delete the layers that change thereby keeping the other attributes of the non changing layers
							cout << "setting up layer:" << layerI << "\n";
							setupLayer(&((*layers)[layerI]), (*widths)[layerI], (layerI == 0) ? 0 : (*widths)[layerI-1]);
							cout << "layer set up:" << layerI << "\n";
						}

					randomise();
					incrementMajorVersion();

					hasChanged = true;
					return SUCCESS;
				}
				else
					return FAILURE;
			}

			status_t	alter(unsigned int layer, layer_modifier mod, bool boolAdd = true)
			/*
			 * Alter a layer within the network. Currently you can only add or remove a bias node from layer zero (the input layer)
			 *
			 * This will randomise the network and increment the major version resetting the minorVerions and revision
			 *
			 */
			{
				if (mod == BIAS_NODE)
					(*layers)[layer].hasBiasNode = boolAdd;
				else
					(*layers)[layer].transition = mod;

                randomise();
                incrementMajorVersion();

                hasChanged = true;

                return SUCCESS;
			}

			status_t alterNode(unsigned int layer, unsigned int node, node_modifier mod)
			{
				(*layers)[layer].nodeInfo->operator[](node).inputType = mod;
				return SUCCESS;
			}

			status_t alterNode(unsigned int layer, unsigned int node, float newP)
			{
				(*layers)[layer].nodeInfo->operator[](node).p = newP;
				return SUCCESS;
			}

			status_t alterNode(unsigned int layer, unsigned int node, bool pIsHalf)
			{
				(*layers)[layer].nodeInfo->operator[](node).pIsOneHalf = pIsHalf;
				return SUCCESS;
			}

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

//				cout << "Layer widths:" << layerWidths->size() << " - " << (*layerWidths)[0] << " " << (*layerWidths)[1] << " " << (*layerWidths)[2] << "\n";

				layers = new vector<layerData>(layerWidths->size());

				setupLayer(&((*layers)[0]), (*layerWidths)[0], 0); 	// arg 2 is the previous layer width therefore 0 for the input layer
				setupLayer(&((*layers)[1]), (*layerWidths)[1], (*layerWidths)[0]);
				setupLayer(&((*layers)[2]), (*layerWidths)[2], (*layerWidths)[1]);

				// create the error vector to be the same length as the output layer
                errorVector = new vector<float>((*layers)[2].nodeCount);	// deleted in the destructor
                randomise();
			}

			void setNodeBias(unsigned int layer, unsigned int node, float bias)
			{
				layers->operator[](layer).nodeInfo->operator[](node).bias = bias;
				cout << "bias layer " << layer << " node " << node  << " bias " << bias << "\n";
			}

			void setLinkWeight(unsigned int layer, unsigned int toNode, unsigned int fromNode, float weight)
			{
				layers->operator[](layer).nodeInfo->operator[](toNode).incomingWeights->operator[](fromNode) = weight;
				cout << "weight layer: " << layer << " from " << fromNode << " to " << toNode << "weight " << weight << "\n";
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

			void setNodeInputType(unsigned int layer, unsigned int node, node_modifier value)
			{
//				cout << "layer: " << layer << " node:" << node << " modifier:" << value << "\n";
				(*layers)[layer].nodeInfo->operator [](node).inputType = value;
			}

			void setNodeP(unsigned int layer, unsigned int node, bool value)
			{
//				cout << "layer: " << layer << " node:" << node << " p is one half:" << value << "\n";
				(*layers)[layer].nodeInfo->operator [](node).pIsOneHalf = value;

			}

			void setNodeP(unsigned int layer, unsigned int node, float value)
			{
//				cout << "layer: " << layer << " node:" << node << " p:" << value << "\n";
				(*layers)[layer].nodeInfo->operator [](node).p = value;

			}

			void resetVersion()
			{
//				cout << "resetting version from " << majorVersion << " " << minorVersion << " " << revision << "\n";
				majorVersion = minorVersion = revision = 0;
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

//            	cout << "layer width: " << width << " prev:" << previousLayerWidth << "\n";

            	layer->nodeCount = width;
            	layer->nodeInfo = new vector<nodeData>(width + 1); // add the space now for the bias node
            	layer->transition = TRANSITION_SIGMOID;
            	layer->hasBiasNode = false;

            	for(nodeI = 0; nodeI < width; nodeI++)
            	{
//            		cout << "adding node: " << nodeI << " content\n";
            		layer->nodeInfo->operator[](nodeI).inputType = INPUT_UNIFORM;
            		layer->nodeInfo->operator[](nodeI).p = 0.5;
            		layer->nodeInfo->operator[](nodeI).pIsOneHalf = true;

            		if (previousLayerWidth != 0)	// previous  == 0 indicates that the layer is the input layer therefore does not need any incoming weights
            		{
            			layer->nodeInfo->operator[](nodeI).incomingWeights = new vector<float>(previousLayerWidth + 1); // add one in case the previous layer has a bias node
//            			cout << nodeI << "< node: space for incoming weights: " << layer->nodeInfo->operator[](nodeI).incomingWeights->size() << "\n";
            		}
            		else
            			; //cout << "no incoming weight space allocated\n";
            	}

            	// and set up the bias node in case it gets switched on later
//        		cout << "adding bias node content\n";
        		layer->nodeInfo->operator[](nodeI).inputType = INPUT_BINARY;
        		layer->nodeInfo->operator[](nodeI).p = 1;
        		layer->nodeInfo->operator[](nodeI).pIsOneHalf = false;
        		layer->nodeInfo->operator[](nodeI).nodeValue = 1;

            }

            void		setup()
            {
				trainingMomentum = 0;
                majorVersion = minorVersion = revision = 0;
                rand_seed = time(NULL);
            }

            void deleteLayers()
            {
            	unsigned int layerI;

            	for (layerI = 0; layerI < layers->size(); layerI++)
            	{
        			if (layerI > 0)
        			{
        				cout << "\tdel layer:" << layerI << "\n";
        				deleteLayer(layerI);
        			}
        			cout << "\tdelete nodeInfo:" << layerI << "\n";
            		delete (*layers)[layerI].nodeInfo;
            	}
            	cout << "delete layers\n";
            	delete layers;
            }

            void deleteLayer(unsigned int layerI)
            {
            	unsigned int nodeI;

            	for (nodeI = 0; nodeI < (*layers)[layerI].nodeCount; nodeI++)
				{
        			cout << "\tdelete layer:" << layerI << " node:" << nodeI << "\n";

        					// this line crashes
        			delete (*layers)[layerI].nodeInfo->operator[](nodeI).incomingWeights;
				}
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
//            			cout << "target node:" << outputI << " input node:" << inputI << " sum :" << sum << "\n";
            		}
            		switch ((*layers)[targetLayer].transition)
            		{
            		case TRANSITION_LINEAR:
            			cout << "Only SIGMOID transitions have been implemented.\n";
            			(*layers)[targetLayer].nodeInfo->operator[](outputI).nodeValue = 1 / (1 + exp((double)(-1 * (sum + (*layers)[targetLayer].nodeInfo->operator[](outputI).bias))));		// sigmoid of the sum plus the bias
            			break;
            		case TRANSITION_BINARY:
            			cout << "Only SIGMOID transitions have been implemented.\n";
            			(*layers)[targetLayer].nodeInfo->operator[](outputI).nodeValue = 1 / (1 + exp((double)(-1 * (sum + (*layers)[targetLayer].nodeInfo->operator[](outputI).bias))));		// sigmoid of the sum plus the bias
            			break;
            		case TRANSITION_SIGMOID:
            			(*layers)[targetLayer].nodeInfo->operator[](outputI).nodeValue = 1 / (1 + exp((double)(-1 * (sum + (*layers)[targetLayer].nodeInfo->operator[](outputI).bias))));		// sigmoid of the sum plus the bias
            			break;
            		case BIAS_NODE:
            			throw format_Error("WTF! the transition is set to BIAS_NODE...");	///////////////////////////////////////////////
            			break;
            		}
//            		cout << "node: " << outputI << " value " << (*layers)[targetLayer].nodeInfo->operator[](outputI).nodeValue << "\n";
            		// from outNode:: (1 / (1 + exp((double)(-1 * biasPlusActivationQuant))))
            	}
//            	cout << "end feedforward \n\n";

            }

            status_t calculateError(vector<float> * desiredOutput)
            {
            	unsigned int i;

            	if (desiredOutput->size() == errorVector->size())
            		for (i=0; i < desiredOutput->size(); i++)
            			(*errorVector)[i] = (*desiredOutput)[i] - (*layers)[2].nodeInfo->operator[](i).nodeValue;
            	else
            		throw internal_Error("Desired vector has the wrong dimensions");  //////////////////////////////////////////////////////

            	return SUCCESS;
            }
			
	private:
    vector<layerData>  * layers;
    float				 trainingLearningRate;
    float				 trainingMomentum;

	// house keeping
	bool				hasChanged;					// set to true after randomisaton or training

	// testing
	vector<float>	*	errorVector;				// pass a pointer to this vector in the test callback
	unsigned int		rand_seed;

	// call backs
	funcTestCallback	testCallback;
	funcRunCallback		runCallback;
	funcTrainCallback	trainCallback;


	// identificaton
	unsigned int		majorVersion;
	unsigned int		minorVersion;
	unsigned int		revision;					// a number to distinguish different versions of the same net
    string				networkName;
};

#endif
