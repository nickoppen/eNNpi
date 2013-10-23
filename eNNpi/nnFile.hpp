
#ifndef _nnfile_h
#define _nnfile_h

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "networkDescription.hpp"
#include <vector>
#include "twoDFloatArray.hpp"
#include "errStruct.hpp"


class NNFile 
// NNFile takes a text file formated by the neural net by the Save or SaveAs functions,
// reads it in completely and then responds to the access calls to pass back each value
{
	public:
                                    NNFile()
                                    {
                                        pFile = NULL;
                                        errMessage = "";
                                    }

                                    NNFile(fstream * theFile)
                                    {
                                        pFile = theFile;
                                        errMessage = "";
                                    }

                                    virtual ~NNFile()
                                    {
                                        // the file is closed and deleted by the calling application
                                    }
		
            status_t				setTo(fstream * theFile)
                                    {
                                        pFile = theFile;
                                        return SUCCESS;
                                    }

            status_t				readInFile()
                                    {
//                                        if (pFile->gcount())		// make sure that the file has something in it
                                            return readInLines();
//                                        else
//                                            throw format_Error(ENN_ERR_NON_FILE);
                                    }
		

	protected:				
            status_t				readInLines()
                                    {
                                        size_t		 strLength = 1;
                                        string		 fragment;
                                        status_t	 decodeResult = SUCCESS;

                                        while (!(pFile->eof()))
                                        {
                                        	getline((*pFile), fragment);
                                        	strLength = fragment.length();

#ifdef _DEBUG_
                                        	cout << "\n" << fragment << "\n";
#endif

                                            if (strLength > 1)
                                                if ((decodeResult = this->decodeLine(&fragment)) != SUCCESS)
                                                    throw format_Error(ENN_ERR_NON_FILE);

                                            fragment.clear();
                                        }
                                        return decodeResult;
                                    }

    virtual	status_t				decodeLine(string * strLines) = 0;
		
	protected:
            unsigned int			nextUIValue(string * fragment, std::string::size_type & startPos, const char limiter = ',')
                                    {
            							std::string::size_type 	endPos;
                                        char	strValue[] = "          ";

                                        endPos = fragment->find(limiter, startPos);	// find the delimiter
                                        if (endPos < startPos)
                                        {
                                            throw format_Error(ENN_ERR_LINE_DECODE_FAILED);
                                        }

                                        fragment->copy(strValue, (endPos - startPos), startPos);
                                        startPos = ++endPos;
                                        return atoi(strValue);
                                    }

            float					nextFValue(string * fragment, std::string::size_type & startPos, const char limiter = ',')
                                    {
            							std::string::size_type	endPos;
                                        char	strValue[] = "                         ";

                                        endPos = fragment->find(limiter, startPos);	// find the delimiter
                                        if (endPos < startPos) // if the delimiter is not found endPos is -1
                                        {
                                            //throw format_Error(ENN_ERR_LINE_DECODE_FAILED + limiter);
                                            throw format_Error(ENN_ERR_LINE_DECODE_FAILED);
                                        }

                                        fragment->copy(strValue, (endPos - startPos), startPos);
                                        startPos = ++endPos;
                                        return (float)atof(strValue);
                                    }

            int						verbArguement(string * line, string & verb, string & arg)
                                    {
                                        // splits verb(args) into verb and (args)
            							std::string::size_type bracketPos;

                                        // pull off the word preceeding the (
                                        bracketPos = line->find('(', 0);
                                        if (bracketPos >= 0)
                                        {
                                        	verb = line->substr(0, bracketPos);
											arg = line->substr(bracketPos);
                                            return 1;
                                        }
                                        return 0;

                                    }

            status_t				decodeNetworkTopology(string * fragment)
                                    {
            							std::string::size_type	startPos = 1;

                                        net.setStandardInputNodes(nextUIValue(fragment, startPos));
                                        net.setHiddenNodes(nextUIValue(fragment, startPos));
                                        net.setOutputNodes(nextUIValue(fragment, startPos, ')'));
#ifdef _DEBUG_
                                      	cout << "Network Topo - InputNodes: " << net.inputNodes() << " HiddenNodes: " << net.hiddenNodes() << " OutputNodes: " << net.outputNodes() << "\n";
#endif

                                        return SUCCESS;
                                    }



			network_description		net;
            fstream	*				pFile;		// temporary storage deleted by the doc
            string					errMessage;
};

#endif
