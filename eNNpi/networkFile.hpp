#ifndef _networkfile_h
#define _networkfile_h

#include "nnFile.hpp"

class networkFile : public NNFile
{
		public:
                        networkFile(fstream * theFile) : NNFile(theFile)
                        {
                            hiddenBiases = NULL;
                            outputBiases = NULL;
                            hasInputBiasNode = false;
                        }

                        networkFile() : NNFile()
                        {
                            hiddenBiases = NULL;
                            outputBiases = NULL;
                            hasInputBiasNode = false;
                        }

                        virtual ~networkFile() //: ~NNFile()
                        {
                            if (hiddenBiases != NULL)
                                delete hiddenBiases;
                            if (outputBiases != NULL)
                                delete outputBiases;
                        }

        void			setTo(fstream * theFile)
                        {
                            if (hiddenBiases != NULL)
                                delete hiddenBiases;
                            if (outputBiases != NULL)
                                delete outputBiases;
                            NNFile::setTo(theFile);
                        }


	// access
		float			linkValue(unsigned int layer, unsigned int node, unsigned int link);
		float			biasValue(unsigned int layer, unsigned int node);

		unsigned int	majorVersion() { return major; }	// major reconstruction of the network (will differ from other major versions by having different inputs etc.
		unsigned int	minorVersion() { return minor; }	// minor versions have different starting point for training
		unsigned int	revision() { return revis; }		// revisions have different amounts of training

        void			networkDescription(network_description * netDes)	// structure passed in by the caller
                        {
                            netDes->setStandardInputNodes(net.standardInputNodes());
                            netDes->setHiddenNodes(net.hiddenNodes());
                            netDes->setOutputNodes(net.outputNodes());
                            netDes->setTrainingLearningRate(net.trainingLearningRate());
                            netDes->setTrainingMomentum(net.trainingMomentum());
 //                           netDes->hasInputLayerBiasNode(net.hasInputLayerBiasNode());
                        }

        void			networkName(string * netName)
                        {
                            (*netName) = name;
                        }

        twoDFloatArray *linkWeights(unsigned int layer)
                        {
                            switch (layer)
                            {
                                case 0:
                                    return &inputLinkWghts;
                                    break;
                                case 1:
                                    return &hiddenLinkWghts;
                                    break;
                                case 2:
                                    throw format_Error(ENN_ERR_LINK_ON_OUTPUT);
                                    break;
                                default:
                                    throw format_Error(ENN_ERR_TOO_MANY_LAYERS);
                            }
                        }

        vector<float> *	nodeBiases(unsigned int layer)
                        {
                            switch (layer)
                            {
                                case 0:
                                    throw format_Error(ENN_ERR_INPUT_NODE_BIAS_REQUESTED);
                                    break;
                                case 1:
                                    return hiddenBiases;
                                    break;
                                case 2:
                                    return outputBiases;
                                    break;
                                default:
                                    throw format_Error(ENN_ERR_TOO_MANY_LAYERS);
                            }
                        }

	private:
        status_t		decodeLine(string * strLine)
                        {
                            string verb = "";
                            string arguements = "";

                            if (verbArguement(strLine, verb, arguements))
                            {
                                if (verb ==  "link")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Link\n";
 #endif

                                    return decodeLink(&arguements);
                                    cout << "done Decode Link - dumping\n";
                                    inputLinkWghts.writeOn(cout);
                                    cout << "Decode Link done with dump - exit\n";
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
                                    hasInputBiasNode = true;
                                    inputLinkWghts.redimension(net.standardInputNodes() + 1, net.hiddenNodes());
                                    return SUCCESS;
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
                            switch (layer)
                            {
                                case 0:
                                    inputLinkWghts.set(node, link, linkWeight);
                                    break;
                                case 1:
                                    hiddenLinkWghts.set(node, link, linkWeight);
                                    break;
                                case 2:
                                    throw format_Error(ENN_ERR_LINK_ON_OUTPUT);
                                    break;	// weights are "owned" by the inNode
                                default:
                                    throw format_Error(ENN_ERR_TOO_MANY_LAYERS);
                            }
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
                            switch (layer)
                            {
                                case 0:
                                    throw format_Error(ENN_ERR_INPUT_NODE_BIAS_REQUESTED);
                                    break;	// actually an error, input nodes have no bias
                                case 1:
                                    hiddenBiases->operator[](node) = nodeBias;
                                    break;
                                case 2:
                                    outputBiases->operator[](node) = nodeBias;
                                    break;
                                default:
                                    throw format_Error(ENN_ERR_TOO_MANY_LAYERS);
                            }

                            return SUCCESS;
                        }

        status_t		decodeName(string * strBracket)
                        {
							std::string::size_type		startPos;
							std::string::size_type		commaPos;

                            // name
                            startPos = 1;	// start at 1 to skip the opening bracket
                            commaPos = strBracket->find(',', startPos);	// find the first comma
                            name = strBracket->substr(1, commaPos - 1);
                            startPos = ++commaPos;

                            major = nextUIValue(strBracket, startPos);
                            minor = nextUIValue(strBracket, startPos);
                            revis = nextUIValue(strBracket, startPos, ')');

#ifdef _DEBUG_
                                        	cout << "Name: " << name << " major-" << major << " minor-" << minor << " revision" << revis << "\n";
#endif

                            return SUCCESS;
                        }

        status_t		decodeNetworkTopology(string * strBracket)
                        {
                            //networkTopology(3,2,2,0.10)

                            status_t returnVal = NNFile::decodeNetworkTopology(strBracket);

                            hiddenBiases = new vector<float>(net.hiddenNodes());
                            outputBiases = new vector<float>(net.outputNodes());
                            inputLinkWghts.dimension(net.inputNodes(), net.hiddenNodes());
                            hiddenLinkWghts.dimension(net.hiddenNodes(), net.outputNodes());
#ifdef _DEBUG_
                                        	cout << "Topo: inputNodes" << net.hiddenNodes() << " hiddenNodes-" << net.hiddenNodes() << " outputNodes-" << net.outputNodes() << "\n";
#endif

                            return returnVal;
                        }

        status_t		decodeVersion(string * strBracket)
                        {
                            return SUCCESS;
                        }

        status_t		decodeLearning(string * strBracket)
                        {
        					std::string::size_type	startPos;

                            startPos = 1;
                            net.setTrainingLearningRate(nextFValue(strBracket, startPos));
                            net.setTrainingMomentum(nextFValue(strBracket, startPos, ')'));

                            return SUCCESS;
                        }

	private:
		unsigned int	major;
		unsigned int	minor;
		unsigned int	revis;

		twoDFloatArray	inputLinkWghts;
		twoDFloatArray	hiddenLinkWghts;
//		twoDFloatArray	outputLinkWghts;
		vector<float> *	hiddenBiases;		// created on readIn deleted at destruction
		vector<float> * outputBiases;		//	"
		bool			hasInputBiasNode;	// true if the input layer has a unaryBiasNode i.e. layerModifier(0, biasNode:true)

        string 			name;
};

#endif
