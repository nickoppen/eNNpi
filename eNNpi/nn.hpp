#ifndef _nn_h
#define _nn_h

#include <sys/stat.h> // POSIX only

#include <sstream>
#include <random>
#include "networkFile.hpp"
#include "dataFile.hpp"
//#include "nnLayer.hpp"

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
                            network_description newNet;

                            newNet.setTrainingLearningRate(learningRateParam);
                            newNet.setNetworkName(newName);

                            newNet.setTrainingMomentum((float)0.0);
                            newNet.setHiddenNodes(hiddenLayerWidth);
                            newNet.setOutputNodes(outputLayerWidth);
                            newNet.setStandardInputNodes(inputLayerWidth);

                            setup(newNet);

                            majorVersion = minorVersion = revision = 0;
                            networkName = newName;

                        }

                        nn(network_description newNet)
                        /*
                         * Create a new network using the data in the object newNet
                         *
                         * Use this call if you want to change default parameters that are not available in the
                         * more primative nn(int, int, int, float, string&) constructor
                         *
                         */
                        {
                        	newNet.setMomentum((float)0.0);

                            setup(newNet);
                            majorVersion = minorVersion = revision = 0;
                            networkName = newNet.networkName();
                        };

                        nn(networkFile * newFile)
                        /*
                         * Reconstruct a network from a saved file with the wrapper newFile
                         */
                        {
                        	setNetworkFile(newFile);
                        };

                        nn(const char * cstrFilename)
                        /*
                         * Reconstruct a network from a file named cstrFilename.
                         *
                         * If you have the filename as a C string use this call. Don't
                         * bother creating a string object to call the nn(string*) constructor
                         * since it will just strip the C string off and call this function
                         *
                         */
                        {
                        	ifstream * pFile;
                        	networkFile * nFile;

                            if (checkExists(cstrFilename))
							{
								pFile = new ifstream(cstrFilename);
								nFile = new networkFile(pFile);
								nFile->readInFile();
								setNetworkFile(nFile);
								delete pFile;
								delete nFile;
							}
                            else
                            	throw format_Error(ENN_ERR_NON_FILE);
                        }

                        nn(string * strFilename)
                        /*
                         * Reconstruct a network from a file named strFilename
                         *
                         * Only use this call if you have the name in a string object already.
                         *
                         */
                        {
                        	nn(strFilename->c_str());
                        }

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
			void		run(const char * cstrFilename, funcRunCallback runComplete = NULL)
			/*
			 * Open the input file named by cstrFilename and run each input vector through the
			 * network, calling the runComplete callback each time.
			 *
			 *	typedef void (*funcRunCallback)(const int index, void * thisNetwork);
			 *	index is the index of the row that has just been run
			 *	theNetwork is a void pointer to this object.
			 *
			 *	Call ((nn*)theNetwork)->runResult(vector<float>* existingVector) to retrieve the result
			 *
			 */
                        {
                            inputFile * inFile;
                            ifstream * pFile;

                            if (checkExists(cstrFilename))
							{
								pFile = new ifstream(cstrFilename);
								inFile = new inputFile(pFile);
								inFile->readInFile();
								run(inFile, runComplete);
								delete inFile;
								delete pFile;
							}
                            else
                            	throw format_Error(ENN_ERR_NON_FILE);
                        }

            void		run(string * strFilename, funcRunCallback runComplete = NULL)
            /*
             * Open the input file named by strFilename and run each input vector throught the
			 * network, calling the runComplete callback each time.
			 *
			 *	typedef void (*funcRunCallback)(const int index, void * thisNetwork);
			 *	index is the index of the row that has just been run
			 *	theNetwork is a void pointer to this object.
			 *
			 *	Call ((nn*)theNetwork)->runResult(vector<float>* existingVector) to retrieve the result
			 *
			 *	Only use this call if you already have the file name as a string object. If you have a
			 *	C string call run(const char * filename, runtimeCallback)
			 *
			 */
                        {
                            run (strFilename->c_str(), runComplete);
                        }

			void		run(inputFile * inFile, funcRunCallback runComplete = NULL)
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
                            unsigned int i;

                            for (i = 0; i < inFile->inputLines(); i++)
                            {
                                run(inFile->inputSet(i), runComplete, i);
                            }

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

//                            theOutputLayer->waitForActivation();									// set up the semaphores
//                            theInputLayer->setInputVector(inputVector);								// set the input nodes with the input vector
//                            theInputLayer->run();													// run the network
//                            theOutputLayer->blockTillValue();										// wait till all the semaphores are cleared

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
//                            theOutputLayer->returnOutputVector(outputVector);		// retrieve the result vector
                            return outputVector;
                        }

            void 		train(string * strFilename, funcTrainCallback trComplete = NULL)
            /*
             * Train the network using the training set in the file called strFilename. Call the trComplete callback once
             * when training is complete.
             *
             * typedef void (*funcTrainCallback)(void * nnObj); passes an anomymous pointer to this object back via the callback
             *
             * call ((nn*)nnObj)->trainingError(vector<float>* existingVector); to retrieve the most recent training error vector
             *
             */
                        {
                            train(strFilename->c_str(), trComplete);
                        }

            void		train(const char * cstrFilename, funcTrainCallback trComplete = NULL)
            /*
             * Train the network using the training set in the file called cstrFilename. Call the trComplete callback once
             * when training is complete.
             *
             * typedef void (*funcTrainCallback)(void * nnObj); passes an anomymous pointer to this object back via the callback
             *
             * call ((nn*)nnObj)->trainingError(vector<float>* existingVector); to retrieve the most recent training error vector
             *
             * If you already have the filename in a string object call train(string *...) otherwise call this function
             *
             */
                        {
                            ifstream * pFile;
                            trainingFile * trFile;

                            if (checkExists(cstrFilename))
							{
								pFile = new ifstream(cstrFilename);
								trFile = new trainingFile(pFile);
								trFile->readInFile();
								train(trFile, trComplete);
								delete pFile;
								delete trFile;
							}
                            else
                            	throw format_Error(ENN_ERR_NON_FILE);

                        }

            void 		train(trainingFile * trFile, funcTrainCallback trComplete = NULL)
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
                            unsigned int i;

                            for (i=0; i < trFile->inputLines(); i++)
                                train(trFile->inputSet(i), trFile->outputSet(i));	// don't pass the call back because we only want it called at the end not after each training set

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
            				if (errorVector->size() != net.outputNodes())
            					return FAILURE;
            				else
            					theOutputLayer->trainingError(errorVector);
							return SUCCESS;
						}

            void		test(const char * cstrTestFilename, funcTestCallback testComplete = NULL)
            /*
             * Run the data component of the training file called the contents of the C string cstrTestFilename and
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
							ifstream * pFile;
							trainingFile * tstFile;

							if (checkExists(cstrTestFilename))
							{
								pFile = new ifstream(cstrTestFilename);
								tstFile = new trainingFile(pFile);
								tstFile->readInFile();
								test(tstFile, testComplete);
								delete pFile;
								delete tstFile;
							}
							else
								throw format_Error(ENN_ERR_NON_FILE);
						}

            void		test(string * strTestFilename, funcTestCallback testComplete = NULL)
            /*
             * Run the data component of the training file called the contents of the string object strTestFilename and
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
             * Note: if you have the filename already as a C string call the function test(const char *... ) rather than this function
             *
             */
						{
							test(strTestFilename->c_str(), testComplete);
						}

            void		test(trainingFile * testFile, funcTestCallback testComplete = NULL)
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
							unsigned int i;

							for (i=0; i < testFile->inputLines(); i++)
								test(i, testFile->inputSet(i), testFile->outputSet(i), testComplete);

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
            				vector<float> outputVec(net.outputNodes());

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
//							theHiddenLayer->randomise(rd);
//							theOutputLayer->randomise(rd);

							hasChanged = true;
							incrementMinorVersion();
						}

	// access
			network_description * networkDescription() { return & net; }	// return the current networ_descripton object
			unsigned int inputNodes() { return net.inputNodes(); }			// return the current number of input nodes (including any input bias node)
			unsigned int hiddenNodes() { return net.hiddenNodes(); }		// return the current number of hidden nodes
			unsigned int outputNodes() { return net.outputNodes(); }		// return the current number of output nodes

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

				ss << "version(1,0,0)\nname(" << networkName << "," << majorVersion << "," << minorVersion << "," << revision << ")\n" ;

				ss << "networkTopology(" << net.standardInputNodes() << "," <<  net.hiddenNodes() <<  "," <<  net.outputNodes() << ")\n";

				ss << "learning(" << net.trainingLearningRate() << "," << net.trainingMomentum() << ")\n";

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
				unsigned int layerNo = 0;

//				delete theInputLayer;
//				delete theHiddenLayer;
//				delete theOutputLayer;

                net.setHiddenNodes(newHidden);
                net.setOutputNodes(newOut);
                net.setStandardInputNodes(newIn);

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

			status_t	alter(network_description * newTopo)
			/*
			 * Alter the topology of the network to be as described in the network_topology object newTopo. Use this call
			 * if you want to alter multiple features at once.
			 *
			 * This will randomise the network and increment the major version resetting the minorVerions and revision
			 *
			 */
			{

//				delete theInputLayer;
//				delete theHiddenLayer;
//				delete theOutputLayer;

				setup(*newTopo);

				randomise();
                incrementMajorVersion();

                hasChanged = true;

                return SUCCESS;
			}

			status_t	alter(unsigned int layer, layer_modifier mod, bool boolAdd = true)
			/*
			 * Alter a layer within the network. Currently you can only add or remove a bias node from layer zero (the input layer)
			 *
			 * This will randomise the network and increment the major version resetting the minorVerions and revision
			 *
			 */
			{
				network_description newNet;

				newNet = net;	// keep all the old values

//				delete theInputLayer;
//				delete theHiddenLayer;
//				delete theOutputLayer;

                newNet.setInputLayerBiasNode(boolAdd);

                setup(newNet);

                randomise();
                incrementMajorVersion();

                hasChanged = true;

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



    // Setup
	private:
            void		incrementRevision() { revision++; }
            void		incrementMinorVersion() { minorVersion++; revision = 0; }
            void		incrementMajorVersion() { majorVersion++; minorVersion = revision = 0; }

            void		setup(network_description newNet)
            {
                unsigned int layerNo = 0;
                net = newNet;
                networkName = net.networkName();

//                theInputLayer = new inputLayer(net, layerNo++);		// deleted in ~nn
//                theHiddenLayer = new hiddenLayer(net, layerNo++);	// deleted in ~nn
//                theOutputLayer = new outputLayer(net, layerNo++);	// deleted in ~nn

//                theInputLayer->connectNodes(theHiddenLayer->nodeList());
//                theHiddenLayer->connectNodes(theOutputLayer->nodeList());

                errorVector = new vector<float>(newNet.outputNodes());	// deleted in the destructor

                randomise();
            }

            void setNetworkFile(networkFile * nFile)
            {
                nFile->networkDescription(&net);

                setup(net);

//                theInputLayer->setLinkWeights(nFile->linkWeights(0));
//                theHiddenLayer->setLinkWeights(nFile->linkWeights(1));
//                theHiddenLayer->setNodeBiases(nFile->nodeBiases(1));
//                theOutputLayer->setNodeBiases(nFile->nodeBiases(2));


                majorVersion = nFile->majorVersion();
                minorVersion = nFile->minorVersion();
                revision = nFile->revision();

                nFile->networkName(&networkName);

                nnNode::setLearningParameters(net.trainingLearningRate(), net.trainingMomentum());

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
			
	private:
//    inputLayer		*	theInputLayer;
//    hiddenLayer		*	theHiddenLayer;
//    outputLayer		*	theOutputLayer;

	network_description	net;
	
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
