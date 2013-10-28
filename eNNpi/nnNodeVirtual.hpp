#ifndef _nnNodeVirtual_h
#define _nnNodeVirtual_h

#include <sstream>
#include "nnLink.hpp"
#include <vector>
#include "nnNodeBase.hpp"
#include "nodeInputType.hpp"
#include "networkDescription.hpp"

#include "nnLink.hpp"

using namespace std ;

//typedef vector<nnLink*> VECTORNNLINKP;

class outNode : public virtual nnNode
{
	public:
                                        outNode(const network_description & net, unsigned int newIndex) //: nnNode(newIndex)
                                        {
                                            // node variables
                                            index = newIndex;
                                            linkCount = 0;
                                            bias = 0.0;
                                            lastBiasChange = 0.0;

                                            // run variables
                                            activationCount = 0;
                                            activationQuantity = 0.0;
                                        }

    virtual								~outNode()	// destroy all the links
                                        {
                                            delete inLinks;

                                        }

						
            void						addInLink(nnLink * link)
                                        {
                                            inLinks->operator[](linkCount++) = link;
                                        }

				
    virtual bool						activationFromLink(float activationLevel)	// called by each link and triggers node value calculation when all links are in
                                        {
                                            activationQuantity += activationLevel;

                                            if (++activationCount == inLinks->size())
                                            {
                                                nnNode::value(outNode::f(bias + activationQuantity));		// pop off the sigmoid function and calculate a new value
                                                activationCount = 0;										// reset the counter for the next run
                                                activationQuantity = 0;										// don't add the next run's values onto the current
                                                return true;												// return true if the nodeValue has been set ie. all the links are in
                                            }
                                            else
                                                return false;
                                        }

            int							randomise(random_device & randSeed, const int input_type = node_input_binary, bool pEqualsOneHalf = true, float p = 0.5)
											// sets all link weights and bias values to a random value and returns it to seed the next call
											// input type and pEqualsOneHalf determines the formula for calculating the range of input values
											// pEqualsOneHalf == true means that the values with have an expected value of 0.5 
                                        {
                                            unsigned int i;
                                            float		 linkWeightVectorLength = (float)0.00; //||w||
                                            nnLink *	 pLink;

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
                                            return 1;
                                        }


			void						setBias(float newBias) { bias = newBias; }	// restore the bias from storage
			
			const size_t				inLinkCount() { return inLinks->size(); }	// return the number of links coming into the node
    virtual	float						adjustBiasReturningOutputError(float learningRate) { return 0.0; };

	protected:
            float						f(float biasPlusActivationQuant)	// implements the activation function f(bias + activationQuantity) = nodeValue
                                        {
                                            if (biasPlusActivationQuant < -50.0)	//>
                                                return 0.0;
                                            else
                                                if (biasPlusActivationQuant > 50.0)
                                                    return 1.0;
                                                else
                                                {
                                                    return (float)(1 / (1 + exp((double)(-1 * biasPlusActivationQuant))));
                                                }
                                        }

            void						setInLinks(vector<nnLink*>	* links)
                                        {
                                            inLinks = links;
                                        }

            vector<nnLink*>			*	getInLinks()
                                        {
                                            return inLinks;
                                        }

            nnLink					*	inLink(unsigned int i)
                                        {
                                            return inLinks->operator[](i);
                                        }

            void						inLink(unsigned int i, nnLink * newLink)
                                        {
                                            inLinks->operator[](i) = newLink;
                                        }

			
            void						storeOn(stringstream * strOut, unsigned int layerNo)
                                        {
                                            (*strOut) << "node(";
                                            (*strOut) << layerNo;
                                            (*strOut) << ",";
                                            (*strOut) << nnNode::index;
                                            (*strOut) << ",";
                                            (*strOut) << bias;
                                            (*strOut) << ")\n";

                                        }

			
	protected:
			unsigned int				activationCount;	// the number of links that have sent activationFromLink messages
			float						activationQuantity;	// the sum of the activation level already recieved
			float						bias;				// added to the total activationQuantity before being run through f()
			float						lastBiasChange;		// lastBiasChange * momentum is added to current bias change
			vector<nnLink*>			*	inLinks;			// output nodes only have inbound links
			vector<nnLink*>::iterator	inLinkI;			// an iterator for reuse 

	private:
			unsigned int				linkCount;
};

class inNode : public virtual nnNode
{
	public:
                                        inNode(const network_description & net, unsigned int newIndex)
                                        {
                                            index = newIndex;
                                            outLinkCount = 0;
                                        }

    virtual								~inNode()	// destroy all the links
                                        {
                                            delete outLinks;
                                        }

	protected:
            void						setOutLinks(vector<nnLink*> * links)
                                        {
                                            outLinks = links;
                                        }

            vector<nnLink*>			*	getOutLinks()
                                        {
                                            return outLinks;
                                        }

            nnLink					*	outLink(unsigned int i)
                                        {
                                            return outLinks->operator[](i);
                                        }

            void						outLink(unsigned int i, nnLink * newLink)
                                        {
                                            // not used
                                            outLinks->operator[](i) = newLink;
                                        }

            void						addOutLink(nnLink * link)
                                        {
                                            outLinks->operator[](outLinkCount++) = link;
                                        }


            nnLink					*	connectTo(outNode * otherNode, unsigned int linkIndex)
                                        {
                                            nnLink * newLink;

                                            newLink = new nnLink((nnNode*)this, (nnNode*)otherNode, linkIndex);	// create a new link from the input node to the hidden node
                                            inNode::addOutLink(newLink);
                                            return newLink;
                                        }

            void						storeOn(stringstream * strOut, unsigned int layerNo)
                                        {

                                            for (outLinkI = outLinks->begin(); outLinkI != outLinks->end(); outLinkI++)
                                            {
                                                (*outLinkI)->storeOn(strOut, layerNo, nnNode::index);
                                            }

                                        }

	// run
	public:
            void						setLinkWeights(vector<float> * weightArray)
                                        {
//            										unsigned int i;
//            										cout << "setting weights for node: " << index << "\n";
//            										for(i=0; i<weightArray->size(); i++)
//            											cout << i << " " << (*weightArray)[i];
//            										cout << "\n";

                                            for (outLinkI = outLinks->begin(); outLinkI != outLinks->end(); outLinkI++)
                                            {
                                                (*outLinkI)->setWeight(weightArray->operator[]((*outLinkI)->linkIndex()));
                                            }
                                        }

            bool						activationFromLink(float activationLevel)	// input links have their value set explicitly before the net is run (should be an error)
                                        {
                                            //throw an error
                                            return false;
                                        }


            void						pushValue()
                                        {
                                            size_t i;
                                            nnLink * theLink;

                                            i = outLinks->size();

                                            for (i = 0; i < outLinks->size(); i++)
                                            {
                                                theLink = outLinks->operator[](i);
                                                theLink->activate(this->value());
                                            }
                                        }


	protected:
			vector<nnLink*>			*	outLinks;
			vector<nnLink*>::iterator	outLinkI;

private:
			unsigned int				outLinkCount;
			
};


#endif
