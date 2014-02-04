#ifndef _networkfile_h
#define _networkfile_h

#include "nnFile.hpp"
#include "nn.hpp"

/*
 * The eNN file wrapper hierarchy:
 *
 *	NNFile
 *		networkFile
 *		dataFile
 *			inputFile
 *			trainingFile
 *
 * networkFile takes a text file formated by the neural net by the SaveTo or SaveOn functions,
 * reads it in completely and then responds to the access calls to pass back each value. The
 * default file extension for a network file is .enn
 *
 * You should not have to edit a .enn file directly yourself.
 */

class networkFile : public NNFile
{
		public:
                        networkFile(ifstream * theFile) : NNFile(theFile)
                        {
                        }

                        networkFile() : NNFile()
                        {
                        }

                        networkFile(const char * cstrFileName) :NNFile(cstrFileName)
                        {

                        }

                        networkFile(string * strFileName) :NNFile(strFileName)
                        {

                        }


                        virtual ~networkFile() //: ~NNFile()
                        {
                        }

        void			setTo(ifstream * theFile)
                        {
                            NNFile::setTo(theFile);
                        }

 virtual nnFileContents	fileType()
						{
							return NETWORK;
						}


	// access
	private:
        status_t		decodeLine(string * strLine)
                        {
                            string verb = "";
                            string arguements = "";

                            if (verbArguement(strLine, verb, arguements))
                            {
                                if (verb ==  "link")
                                {
                                	status_t rVal;
#ifdef _DEBUG_
                                        	cout << "Decode Link\n";
 #endif

                                    rVal = decodeLink(&arguements);
                                    return rVal;
                                }
                                if (verb == "nodeModifier")
                                {
#ifdef _DEBUG_
                                	cout << "Decode node modifier\n";
#endif
                                	return decodeNodeModifier(&arguements);
                                }
                                if (verb ==  "node")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode node\n";
#endif
                                    return decodeNode(&arguements);
                                }
                                if (verb == "version")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode version\n";
#endif
                                    // no need to check the file version just yet
                                    return SUCCESS;
                                }
                                if (verb == "name")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Name\n";
#endif
                                    return decodeName(&arguements);
                                }
                                if (verb ==  "networkTopology")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Topo\n";
#endif
                                    return decodeNetworkTopology(&arguements);
                                }
                                if (verb == "comment")
                                {
                                    // do nothing with comments
                                    return SUCCESS;
                                }
                                if (verb == "learning")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Learning\n";
#endif
                                    return decodeLearning(&arguements);
                                }
                                if (verb == "layerModifier")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Layer mod\n";
#endif
                                        	// need to actually decode the layerModifer clause
                                    return decodeLayerModifier(&arguements);
                                }

                                errMessage = ENN_ERR_UNK_KEY_WORD;
                                errMessage += ": ";
                                errMessage += verb.c_str();
                                throw format_Error(errMessage.c_str());
                            }
                            throw format_Error(ENN_ERR_NON_FILE);
                        }

        status_t		decodeLink(string * strBracket)
                        {
                            unsigned int layer;
                            unsigned int node;
                            unsigned int link;
                            float		 linkWeight;

                            std::string::size_type		 startPos;

                            startPos = 1;
                            layer = nextUIValue(strBracket, startPos);
                            node = nextUIValue(strBracket, startPos);
                            link = nextUIValue(strBracket, startPos);
                            linkWeight = nextFValue(strBracket, startPos, ')');

#ifdef _DEBUG_
                        	cout << "Link: layer-" << layer << " innode-" << node << " outnode-" << link << " weight: " << linkWeight << "\n";
#endif
                              ((nn*)theNetwork)->setLinkWeight(layer, node, link, linkWeight);

                            return SUCCESS;
                        }

        status_t		decodeNodeModifier(string * strBracket)
						{
							unsigned int layer;
							unsigned int node;
        					string modifier = "";
        					string value = "";
                            std::string::size_type startPos = 1;

                            layer = nextUIValue(strBracket, startPos);
                            node = nextUIValue(strBracket, startPos);

#ifdef _DEBUG_
                            cout << "Modifiers on layer:" << layer << " node: " << node << " -";
#endif
                            while (startPos != 0)
                            {
            					keyValue(strBracket, startPos, modifier, value);
            					if (modifier == "input")
            					{
									if (value == "UNIFORM")
										((nn*)theNetwork)->setNodeInputType(layer, node, INPUT_UNIFORM);
									else
										if (value == "BINARY")
											((nn*)theNetwork)->setNodeInputType(layer, node, INPUT_BINARY);
										else
											if (value == "BIPOLAR")
												((nn*)theNetwork)->setNodeInputType(layer, node, INPUT_BIPOLAR);
											else
											{
												cout << value;
												throw format_Error(ENN_ERR_UNKNOWN_NODE_MODIFIER_VALUE);//, value.c_str());
											}
#ifdef _DEBUG_
									cout << " input type:" << value;
#endif
            					}
            					else
            						if (modifier == "pIsOneHalf")
            						{
    									if (value == "true")
    										((nn*)theNetwork)->setNodeP(layer, node, true);
    									else
    										((nn*)theNetwork)->setNodeP(layer, node, false);
#ifdef _DEBUG_
    									cout << " p is one half:" << value;
#endif
            						}
            						else
            							if (modifier == "p")
            							{
            								((nn*)theNetwork)->setNodeP(layer, node, (float)atof(value.c_str()));
#ifdef _DEBUG_
            								cout << " p:" << value;
#endif
            							}
            							else
            							{
            								cout << modifier << "\n";
            								throw format_Error((modifier + "< " + ENN_ERR_UNKNOWN_NODE_MODIFIER).c_str());//, value.c_str());
            							}

                            }
#ifdef _DEBUG_
                            cout << "\n";
#endif
                            return SUCCESS;
						}

        status_t		decodeNode(string * strBracket)
                        {
                            unsigned int layer;
                            unsigned int node;
                            float		 nodeBias;

                            std::string::size_type		 startPos;

                            startPos = 1;
                            layer = nextUIValue(strBracket, startPos);
                            node = nextUIValue(strBracket, startPos);
                            nodeBias = nextFValue(strBracket, startPos, ')');

#ifdef _DEBUG_
                                        	cout << "Node: layer-" << layer << " node-" << node << " bias-" << nodeBias << "\n";
#endif

                              ((nn*)theNetwork)->setNodeBias(layer, node, nodeBias);

                            return SUCCESS;
                        }

        status_t		decodeName(string * strBracket)
                        {
							std::string::size_type		startPos;
							std::string::size_type		commaPos;
							std::string					name;

							unsigned int major;
							unsigned int minor;
							unsigned int revis;

                            // name
                            startPos = 1;	// start at 1 to skip the opening bracket
                            commaPos = strBracket->find(',', startPos);	// find the first comma
                            name = strBracket->substr(1, commaPos - 1);
                            startPos = ++commaPos;
                            ((nn*)theNetwork)->setName(&name);

                            major = nextUIValue(strBracket, startPos);
                            minor = nextUIValue(strBracket, startPos);
                            revis = nextUIValue(strBracket, startPos, ')');
                            ((nn*)theNetwork)->setVersion(major, minor, revis);

#ifdef _DEBUG_
                                        	cout << "Name: " << name << " major-" << major << " minor-" << minor << " revision" << revis << "\n";
#endif

                            return SUCCESS;
                        }

        status_t		decodeNetworkTopology(string * strBracket)
                        {
        					vector<unsigned int> layerWidths(maxLayers);

                            status_t returnVal = NNFile::decodeNetworkTopology(strBracket, maxLayers, &layerWidths);
                            ((nn*) theNetwork)->setNetworkTopology(&layerWidths);

                            return returnVal;
                        }

        status_t		decodeVersion(string * strBracket)
                        {
        					string curVer(ennVersion);
        					if (*strBracket == curVer)
        						return SUCCESS;
        					else
        						throw format_Error(ENN_ERR_UNSUPPORTED_ENN_FILE_FORMAT);

        					return FAILURE;		// just to keep the formatter happy
                        }

        status_t		decodeLearning(string * strBracket)
                        {
        					std::string::size_type	startPos;

                            startPos = 1;
                            ((nn*)theNetwork)->setTrainingLearningRate(nextFValue(strBracket, startPos));
                            ((nn*)theNetwork)->setTrainingMomentum(nextFValue(strBracket, startPos, ')'));

                            return SUCCESS;
                        }

        status_t		decodeLayerModifier(string * strBracket)
						{
        					unsigned int whichLayer;
        					string modifier = "";
        					string value = "";
        					std::string::size_type	startPos = 1;

        					whichLayer = nextUIValue(strBracket, startPos);
        					keyValue(strBracket, startPos, modifier, value);

        					if (modifier == "biasNode")
        					{
								if (whichLayer < 3)	// expand to include all but output layer
								{
									if (value == "true")
										((nn*)theNetwork)->setHasBiasNode(whichLayer, true);
									else
										((nn*)theNetwork)->setHasBiasNode(whichLayer, false);
								}
								else
									throw format_Error(ENN_ERR_BIAS_NODE_ON_INVALID_LAYER);
        					}
        					else
        						throw format_Error(ENN_ERR_UNK_MODIFIER);

        					return SUCCESS;

						}

        status_t		readInLines(bool shouldNotBeHere)
						{
							return FAILURE;
						}

	private:
        /*
         * All data is passed to the network as it is read in
         */
};

#endif
