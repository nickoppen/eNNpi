#ifndef _nn_h
#define _nn_h

#include <sys/stat.h> // POSIX only

#include <sstream>
#include <random>
#include "networkFile.hpp"
#include "dataFile.hpp"
#include "nnLayer.hpp"

typedef void (*funcRunCallback)(const int, void *);
typedef void (*funcTrainCallback)(void *);
typedef void (*funcTestCallback)(vector<float>*, void *);

class nn
{
	public:
						nn(int inputLayerWidth, int hiddenLayerWidth, int outputLayerWidth, float learningRateParam, string & newName)
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
                        {
                        	newNet.setMomentum((float)0.0);

                            setup(newNet);
                            majorVersion = minorVersion = revision = 0;
                            networkName = newNet.networkName();
                        };

                        nn(networkFile * newFile)
                        {
                        	setNetworkFile(newFile);
                        };

                        nn(const char * cstrFilename)
                        {
                        	ifstream * pFile;
                        	networkFile * nFile;

                            if (checkExists(cstrFilename))
							{
								pFile = new ifstream(cstrFilename);
								nFile = new networkFile(pFile);
								nFile->readInFile();
								setNetworkFile(nFile);
							}
                            else
                            	throw format_Error(ENN_ERR_NON_FILE);
                        }

                        nn(string * strFilename)
                        {
                        	nn(strFilename->c_str());
                        }

                        ~nn()
                        {
                        	delete errorVector;
                            delete theInputLayer;
                            delete theHiddenLayer;
                            delete theOutputLayer;
                        }

						
	// operation
            void		run(string * strFilename, funcRunCallback runComplete = NULL)
                        {
                            run (strFilename->c_str(), runComplete);
                        }

			void		run(const char * cstrFilename, funcRunCallback runComplete = NULL)
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

			void		run(inputFile * inFile, funcRunCallback runComplete = NULL)		// call back because running is asychronous
                        {
                            // call run(vector) from inFile
                            unsigned int i;

                            for (i = 0; i < inFile->inputLines(); i++)
                            {
                                run(inFile->inputSet(i), runComplete, i);
                            }

                        }
            void		run(vector<float> * inputVector, funcRunCallback runComplete = NULL, const int index = 0)
                        {
//                            cout << "running:" << (*inputVector)[0] << "\n";

                            theOutputLayer->waitForActivation();									// set up the semaphores
                            theInputLayer->setInputVector(inputVector);								// set the input nodes with the input vector
                            theInputLayer->run();													// run the network
                            theOutputLayer->blockTillValue();										// wait till all the semaphores are cleared

                            if (runComplete != NULL)
                                runComplete(index, (void*)this);
                        }

            void		run(vector<float> * inputVector, vector<float> * outputVector)	/// synchronous call
                        {
                            run(inputVector);
                            // wait for the result
                            outputVector = runResult();
                        }

            vector<float> * runResult()
                        {
                            vector<float> * outputVector;
                            outputVector = new vector<float>(net.outputNodes());
                            theOutputLayer->returnOutputVector(outputVector);		// retrieve the result vector
                            return outputVector;
                        }

            void 		train(string * strFilename, funcTrainCallback trComplete = NULL)
                        {
                            train(strFilename->c_str(), trComplete);
                        }

            void		train(const char * cstrFilename, funcTrainCallback trComplete = NULL)
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
                        {
                            // train
                            try
                            {
                                // run
                                run(inputVector);

//                                cout << "train: " << (*inputVector)[1] << "\n";

                                theOutputLayer->setDesiredValues(desiredVector);

                                theHiddenLayer->waitForTraining();
                                theHiddenLayer->train();
                                theHiddenLayer->blockTillTrained();

                                theInputLayer->waitForTraining();
                                theInputLayer->train();
                                theInputLayer->blockTillTrained();

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
            			// the errorVector must exist and be the right size
						{
            				if (errorVector->size() != net.outputNodes())
            					return FAILURE;
            				else
            					theOutputLayer->trainingError(errorVector);
							return SUCCESS;
						}

        	// test
            void		test(string * strTestFilename, funcTestCallback testComplete = NULL)
						{
							test(strTestFilename->c_str(), testComplete);
						}

            void		test(const char * cstrTestFilename, funcTestCallback testComplete = NULL)
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

            void		test(trainingFile * testFile, funcTestCallback testComplete = NULL)
						{
							unsigned int i;

							for (i=0; i < testFile->inputLines(); i++)
								test(testFile->inputSet(i), testFile->outputSet(i), testComplete);

						}

            void		test(vector<float> * inputVector, vector<float> * desiredOutput, funcTestCallback testComplete = NULL)
						{
            				size_t i;
							// run
							run(inputVector);

							// block til value

							// compare
							theOutputLayer->returnOutputVector(errorVector);

							for (i = 0; i != errorVector->size(); i++)
								(*errorVector)[i] = (*errorVector)[i] - (*desiredOutput)[i];

							if (testComplete != NULL)
								testComplete(errorVector, (void*)this);
						}

            void		randomise()
						{
							random_device rd;
							theHiddenLayer->randomise(rd);
							theOutputLayer->randomise(rd);

							hasChanged = true;
							incrementMinorVersion();
						}

	// access
			network_description * networkDescription() { return & net; }
			unsigned int inputNodes() { return net.inputNodes(); }
			unsigned int hiddenNodes() { return net.hiddenNodes(); }
			unsigned int outputNodes() { return net.outputNodes(); }

			status_t 	saveTo(string * strPath)
			{
				return saveTo(strPath->c_str());
			}

			status_t	saveTo(const char * cstrPath)
			{
				fstream * pFile;
				status_t rVal;
				char cstrPathFile[] = "                                        ";	// dumb
				char cstrFileName[] = "                                ";			// dumb

				if (checkExists(cstrPath, false))
				{
					sprintf(cstrPathFile, "%s//%s", cstrPath, defaultName(cstrFileName));
//					cout << "saving to <" << cstrPathFile << "> " << majorVersion << " " << minorVersion << " " << revision << "\n";

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
			{
				string strContent;
				status_t rVal;
				rVal = saveOn(&strContent);
				(*pFile) << strContent;

				return rVal;
			}

	// Modify
			status_t	alter(int newIn, int newHidden, int newOut)
			{
				unsigned int layerNo = 0;

				delete theInputLayer;
				delete theHiddenLayer;
				delete theOutputLayer;

                net.setHiddenNodes(newHidden);
                net.setOutputNodes(newOut);
                net.setStandardInputNodes(newIn);

                theInputLayer = new inputLayer(net, layerNo++);
                theHiddenLayer = new hiddenLayer(net, layerNo++);
                theOutputLayer = new outputLayer(net, layerNo++);

                theInputLayer->connectNodes(theHiddenLayer->nodeList());
                theHiddenLayer->connectNodes(theOutputLayer->nodeList());

                randomise();
                incrementMajorVersion();

				return SUCCESS;
			}

			status_t	alter(network_description * newTopo)
			{

				delete theInputLayer;
				delete theHiddenLayer;
				delete theOutputLayer;

				setup(*newTopo);

				randomise();
                incrementMajorVersion();

                return SUCCESS;
			}

			status_t	alter(unsigned int layer, layer_modifier mod, bool boolAdd = true)
			{
				network_description newNet;

				newNet = net;	// keep all the old values

				delete theInputLayer;
				delete theHiddenLayer;
				delete theOutputLayer;

                newNet.setInputLayerBiasNode(boolAdd);

                setup(newNet);

                randomise();
                incrementMajorVersion();

                return SUCCESS;
			}

	// save to disk
            status_t	saveOn(string * strOut)	// save the net in the given existing string
            {
            	stringstream ss;

            	ss.precision(8);

                ss << "version(1,0,0)\nname(" << networkName << "," << majorVersion << "," << minorVersion << "," << revision << ")\n" ;

                ss << "networkTopology(" << net.standardInputNodes() << "," <<  net.hiddenNodes() <<  "," <<  net.outputNodes() << ")\n";

                ss << "learning(" << net.trainingLearningRate() << "," << net.trainingMomentum() << ")\n";

                // call the detail storage process here
                theInputLayer->storeOn(&ss);
                theHiddenLayer->storeOn(&ss);
                theOutputLayer->storeOn(&ss);

                hasChanged = false;

                (*strOut) = ss.str();

                return SUCCESS;
            }

            char *	defaultName(char * buffer)
            // the calling function must make sure that there is enough room in the buffer
            {
            	sprintf(buffer, "%s_%d_%d_%d.enn", networkName.c_str(), majorVersion, minorVersion, revision);

            	return buffer;
            }

			bool		needsSaving() { return hasChanged; }



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

                theInputLayer = new inputLayer(net, layerNo++);
                theHiddenLayer = new hiddenLayer(net, layerNo++);
                theOutputLayer = new outputLayer(net, layerNo++);

                theInputLayer->connectNodes(theHiddenLayer->nodeList());
                theHiddenLayer->connectNodes(theOutputLayer->nodeList());

                errorVector = new vector<float>(newNet.outputNodes());	// deleted in the destructor

                randomise();
            }

            void setNetworkFile(networkFile * nFile)
            {
                nFile->networkDescription(&net);

                setup(net);

                theInputLayer->setLinkWeights(nFile->linkWeights(0));
                theHiddenLayer->setLinkWeights(nFile->linkWeights(1));
                theHiddenLayer->setNodeBiases(nFile->nodeBiases(1));
                theOutputLayer->setNodeBiases(nFile->nodeBiases(2));


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
    inputLayer		*	theInputLayer;
    hiddenLayer		*	theHiddenLayer;
    outputLayer		*	theOutputLayer;

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
