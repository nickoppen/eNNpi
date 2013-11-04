#ifndef _datafile_h
#define _datafile_h

#include "nnFile.hpp"


/*
 *	NNFile
 *		networkFile
 *		dataFile
 *			inputFile
 *			trainingFile
 */
 
class dataFile :  public NNFile
{
	public:
                    dataFile() : NNFile()
                    {
                        lineCount = 0;
                        inputArray = NULL;
                    }

                    dataFile(ifstream * theFile) : NNFile(theFile)
                    {
                        lineCount = 0;
                        inputArray = NULL;
                    }

                    virtual ~dataFile() //: ~NNFile()
                    {
                        if (inputArray != NULL)
                            delete inputArray;
                    }

	
	public:		// access
                    unsigned int			inputLines()	// return how many lines have been read in
                                            {	// return how many lines have been read in
                                                unsigned int fileLen;
                                                unsigned int inputVecWidth;

                                                if (inputArray->dimensions(fileLen, inputVecWidth))
                                                    return fileLen;
                                                else
                                                    return 0;

                                            }

                    vector<float> *			inputSet(unsigned int row)
                                            {
                                                return inputArray->values(row);
                                            }

					network_description *	networkDescription() { return &net; }

	private:
    virtual			status_t				decodeLine(string * strLine) = 0;

	protected:
					unsigned int			lineCount;
					twoDFloatArray	*		inputArray;
	
};

class inputFile : public dataFile
{
	public:
                    inputFile() : dataFile() { }
                    inputFile(ifstream * theFile) : dataFile(theFile) { }
                    virtual ~inputFile() // : ~dataFile()
                    { }
	
	private:
    status_t		decodeLine(string * strLine)
                    {
    					std::string::size_type			bracketPos;
                        string							verb = "";
                        string							arguements = "";

                        bracketPos = strLine->find('(', 0);
                        if (bracketPos == std::string::npos)
                            throw format_Error(ENN_ERR_NON_FILE);

                        if (verbArguement(strLine, verb, arguements))
                        {
                            if (verb ==  "inputVector")
                            {
                                if (lineCount++ == 0)
                                    inputArray = new twoDFloatArray(net.standardInputNodes());
                                else
                                    inputArray->addRow();
#ifdef _DEBUG_
                                        	cout << "Decoding Input Vector\n";
#endif


                                return decodeInputVector(&arguements, inputArray->values(lineCount - 1));
                            }
                            if (verb ==  "networkTopology")
                            {
#ifdef _DEBUG_
                                        	cout << "Decode Topology\n";
#endif

                                return decodeNetworkTopology(&arguements);
                            }
//                            errMessage.Format("%s: %s", ENN_ERR_UNK_KEY_WORD, verb.GetBuffer());
                            errMessage = ENN_ERR_UNK_KEY_WORD;
                            errMessage += ": ";
                            errMessage += verb;
                            throw format_Error(errMessage.c_str());
                        }
                        else
                            throw format_Error(ENN_ERR_NON_FILE);

                        return FAILURE; // will not happen
                    }

    status_t		decodeInputVector(string * fragment, vector<float> * lineVector)
                    {
                        float		 inputValue;
                        unsigned int node;
                        std::string::size_type		 startPos;

                        startPos = 1;

#ifdef _DEBUG_
                                        	cout << "Input Values,";
#endif

                        for (node = 0; node < (net.standardInputNodes() - 1); node++)	//>
                        {
                            inputValue = nextFValue(fragment, startPos);
                            lineVector->operator[](node) = inputValue;
#ifdef _DEBUG_
                                        	cout << " Node " << node << ": " << inputValue;
#endif
                        }
                        inputValue = nextFValue(fragment, startPos, ')');
#ifdef _DEBUG_
                                        	cout << " Node " << node << ": " << inputValue << "\n";
#endif
                        lineVector->operator[](node) = inputValue;

                        return SUCCESS;
                    }


};

class trainingFile : public dataFile
{
	public:
                    trainingFile() : dataFile()
                    {
                        outputArray = NULL;
                    }

                    trainingFile(ifstream * theFile) : dataFile(theFile)
                    {
                        outputArray = NULL;
                    }

                    virtual ~trainingFile() //: ~dataFile()
                    {
                        if (outputArray != NULL)
                            delete outputArray;
                    }


                    vector<float> * outputSet(unsigned int row)
                    {
                        return outputArray->values(row);
                    }

                    vector<float> * outputVector(unsigned int row)
					{
						return outputArray->values(row);
					}


	private:
    status_t		decodeLine(string * strLine)
                    {
                        std::string::size_type				bracketPos;
                        string								verb = "";
                        string								arguements = "";

                        bracketPos = strLine->find('(', 0);
                        if (bracketPos == std::string::npos) throw format_Error(ENN_ERR_NON_FILE);

                        if (verbArguement(strLine, verb, arguements))
                        {
                            if (verb ==  "networkTopology")
                            {
#ifdef _DEBUG_
                                        	cout << "Decode Topology\n";
#endif
                                return decodeNetworkTopology(&arguements);
                            }
                            if (verb ==  "inputOutputVector")
                            {
                                if (lineCount++ == 0)
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Input/Output Vector\n";
#endif
                                    inputArray = new twoDFloatArray(net.standardInputNodes());
                                    outputArray = new twoDFloatArray(net.outputNodes());
                                }
                                else
                                {
                                    inputArray->addRow();
                                    outputArray->addRow();
                                }

                                return decodeTrainingVector(&arguements, inputArray->values(lineCount -1), outputArray->values(lineCount - 1));
                            }
                            errMessage = ENN_ERR_UNK_KEY_WORD;
                            errMessage += ": ";
                            errMessage += verb;
                            throw format_Error(errMessage.c_str());

                        }
                        else
                            throw format_Error(ENN_ERR_NON_FILE);

                        return FAILURE; // will not happen
                    }

    status_t		decodeTrainingVector(string * fragment, vector<float> * inVector, vector<float> * outVector)
                    {
                        float		 readValue;
                        unsigned int node;
                        std::string::size_type		 startPos;

                        startPos = 1;

#ifdef _DEBUG_
                                    	cout << "Input Vector,";
#endif
                        for (node = 0; node < (net.standardInputNodes() - 1); node++)	//>
                        {
                            readValue = nextFValue(fragment, startPos);
                            inVector->operator[](node) = readValue;
#ifdef _DEBUG_
                                      	cout << " Node: " << node << ": " << readValue;
#endif
                        }
                        readValue = nextFValue(fragment, startPos, ';');
#ifdef _DEBUG_
                                      	cout << " Node: " << node << ": " << readValue << "\n";
#endif
                        inVector->operator[](node) = readValue;

#ifdef _DEBUG_
                                      	cout << "Output Vector,";
#endif
                        for (node = 0; node < (net.outputNodes() - 1); node++)		//>
                        {
                            readValue = nextFValue(fragment, startPos);
                            outVector->operator[](node) = readValue;
#ifdef _DEBUG_
                                      	cout << " Node: " << node << ": " << readValue;
#endif
                        }
                        readValue = nextFValue(fragment, startPos, ')');
#ifdef _DEBUG_
                                      	cout << " Node: " << node << ": " << readValue << "\n";
#endif
                        outVector->operator[](node) = readValue;

                        return SUCCESS;
                    }
	
	twoDFloatArray * outputArray;
};

#endif
