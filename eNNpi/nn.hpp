#ifndef _nn_h
#define _nn_h

#include <sstream>
#include <random>
#include "networkFile.hpp"
#include "dataFile.hpp"
#include "nnLayer.hpp"

typedef void (*funcRunCallback)(const int, void *);
typedef void (*funcTrainCallback)(void *);

class nn
{
	public:
						nn(int inputLayerWidth, int hiddenLayerWidth, int outputLayerWidth, float learningRateParam, string & newName)
                        {
                            network_description newNet;

                            newNet.setHiddenNodes(hiddenLayerWidth);
                            newNet.setOutputNodes(outputLayerWidth);
                            newNet.setStandardInputNodes(inputLayerWidth);
                            newNet.setTrainingLearningRate(learningRateParam);
                            newNet.setNetworkName(newName);

//                            newNet.hasInputLayerBiasNode(true);	// temporary
                            newNet.setTrainingMomentum((float)0.0);

                            setup(newNet);

                            majorVersion = minorVersion = revision = 0;
                            networkName = newName;

                        }

                        nn(network_description newNet)
                        {
                        //	newNet.inputLayerBiasNode = true;	// temporary
                        //	newNet.momentum = (float)0.0;

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
                        	fstream * pFile;
                        	networkFile * nFile;

    						pFile = new fstream(cstrFilename);
    						nFile = new networkFile(pFile);
    						nFile->readInFile();
    						setNetworkFile(nFile);

                        }

                        nn(string * strFilename)
                        {
                        	nn(strFilename->c_str());
                        }

                        ~nn()
                        {
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
                            fstream * pFile;

                            pFile = new fstream(cstrFilename);
                            inFile = new inputFile(pFile);
                            inFile->readInFile();
                            run(inFile, runComplete);
                            delete inFile;
                            delete pFile;
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
                            cout << "running:" << (*inputVector)[0] << "\n";

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
                            fstream * pFile;
                            trainingFile * trFile;

                            pFile = new fstream(cstrFilename);
                            trFile = new trainingFile(pFile);
                            trFile->readInFile();
                            train(trFile, trComplete);
                            delete pFile;
                            delete trFile;

                        }

            void 		train(trainingFile * trFile, funcTrainCallback trComplete = NULL)
                        {
                            // call train with each vector
                            unsigned int i;

                            for (i=0; i < trFile->inputLines(); i++)
                                train(trFile->inputSet(i), trFile->outputSet(i));	// don't pass the call back because we only want it called at the end not after each training set

                            if (trComplete != NULL)
                                trComplete((void*)this);
                        }

            void		train(vector<float> * inputVector, vector<float> * desiredVector, funcTrainCallback trComplete = NULL)
                        {
                            // run
                            run(inputVector);

                            // train
                            try
                            {
                                cout << "train: " << (*inputVector)[1] << "\n";

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
                            revision++;
                        }

            vector<float> *	trainingError()
            {
                vector<float> * errorVector;
                errorVector = new vector<float>(net.outputNodes());	// leave empty for now
                theOutputLayer->trainingError(errorVector);
                return errorVector;
            }

            void		test(vector<float> * inputVector, vector<float> * desiredOutput, vector<float> * errorVector)
            {
                // run
                run(inputVector);

                // compare
                theOutputLayer->setDesiredValues(desiredOutput);
                errorVector = trainingError();

            }

            void		randomise()
            {
                random_device rd;
                theHiddenLayer->randomise(rd);
                theOutputLayer->randomise(rd);

                hasChanged = true;
                minorVersion++;
                revision = 0;
            }

	// access
			network_description * networkDescription() { return & net; }

			status_t 	saveTo(string * strFile)
			{
				return saveTo(strFile->c_str());
			}

			status_t	saveTo(const char * cstrFile)
			{
				fstream * pFile;
				status_t rVal;

				pFile = new fstream();
				pFile->open(cstrFile, ios::out);
				rVal = saveTo(pFile);
				pFile->close();
				delete pFile;

				return rVal;
			}

			status_t	saveTo(fstream * pFile)
			{
				string strContent;
				status_t rVal;
				rVal = saveOn(&strContent);
				(*pFile) << strContent;

				return rVal;
			}

	// save to disk
            status_t	saveOn(string * strOut)	// save the net in the given existing string
            {
            	stringstream ss;

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

            void		defaultName(string & name)
            {
            	stringstream ss;

            	ss << "Network-" << majorVersion <<  "-" << minorVersion << "-" << revision;
            	name = ss.str();
            }

			bool		needsSaving() { return hasChanged; }
	// test
            void		testNN()
            {
                    // not implemented yet

            }

            void		incrementRevision() { revision++; }
            void		incrementMinorVersion() { minorVersion++; revision = 0; }
            void		incrementMajorVersion() { majorVersion++; minorVersion = revision = 0; }


    // Setup
	private:
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

                randomise();
            }

            void setNetworkFile(networkFile * nFile)
            {
                nFile->networkDescription(&net);

            //	net.inputLayerBiasNode = true;	// temporary
            //	net.momentum = (float)0.0;

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
			
	private:
    inputLayer		*	theInputLayer;
    hiddenLayer		*	theHiddenLayer;
    outputLayer		*	theOutputLayer;

	network_description	net;
	
	// house keeping
	bool				hasChanged;					// set to true after randomisaton or training

	// identificaton
	unsigned int		majorVersion;
	unsigned int		minorVersion;
	unsigned int		revision;					// a number to distinguish different versions of the same net
    string				networkName;
};

#endif
