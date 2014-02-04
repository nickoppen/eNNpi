
#include "nn.hpp"
#include "networkFile.hpp"
#include "dataFile.hpp"

void callback_RunComplete(const int index, void * caller)
{
	vector<float> * resVec;	// created by the net
	unsigned int i;

	resVec = new vector<float>(((nn*)caller)->layerNWidth());
	resVec = ((nn*)caller)->runResult(resVec);
	cout << index;
	for(i = 0; i< resVec->size(); i++)
		cout << "," << i << "," << (*resVec)[i];
	cout << "\n";
	delete resVec;
}

void callback_TrainingComplete(void * caller)
{
	vector<float> * errVec;	// created by the net
	unsigned int i;

	errVec = new vector<float>(((nn*)caller)->layerNWidth());

	if ( ((nn*)caller)->trainingError(errVec) == SUCCESS)
	{
		cout << "Last Training Error Vector -";
		for(i = 0; i< errVec->size(); i++)
			cout << " " << i << ": " << (*errVec)[i];
		cout << "\n";
	}
	else
		cout << "Error: failed to retrieve the error vector.\n";

	delete errVec;
}

void callback_TestComplete(const int index, vector<float>* inputVector, vector<float>* desiredOutput, vector<float>* actualOutput, vector<float> * errVec, void * caller)
{
	unsigned int i;

	if (index == 0)		// title row
	{
		cout << "Index";
		for(i = 0; i< inputVector->size(); i++)
			cout << "\tInput:" << i;
		for(i = 0; i< desiredOutput->size(); i++)
			cout << "\tDesired:" << i;
		for(i = 0; i< actualOutput->size(); i++)
			cout << "\tActual:" << i;
		for(i = 0; i< errVec->size(); i++)
			cout << "\tError:" << i;
		cout << "\n";
	}

	cout << index;
	for(i = 0; i< inputVector->size(); i++)
		cout << "\t" << (*inputVector)[i];
	for(i = 0; i< desiredOutput->size(); i++)
		cout << "\t" << (*desiredOutput)[i];
	for(i = 0; i< actualOutput->size(); i++)
		cout << "\t" << (*actualOutput)[i];
	for(i = 0; i< errVec->size(); i++)
		cout << "\t" << (*errVec)[i];
	cout << "\n";
}

int main(int argc, char *argv[])
{
	nn * theNet = NULL;
	string argvI;
	string strArg;
	string strSave;
	int in, out, hidden;
	float fVal;
	bool unknownFlag = false;
	bool quiet = false;

	int i;

	for (i=1; i<argc; i++)
	{
		argvI = argv[i];
		if (argvI == "-n")
		{
				if (theNet != NULL)
				{
					cout << "A network was already loaded and has been deleted.\n";
					delete theNet;
				}

				try
				{
					networkFile netFile(argv[++i]);
					theNet = new nn(&netFile);

					if (!quiet)
						cout << "Done with -n\n";

				}
				catch (format_Error & e)
				{
					cout << e.mesg << "\n";
				}
		}
		else

			if (argvI == "-t")
			{
				if (theNet == NULL)
					cout << "A network must be loaded before it can be trained.\n";
				else
					try
					{
						trainingFile trFile(argv[++i]);
						theNet->train(&trFile, &callback_TrainingComplete);

						if (!quiet)
							cout << "Done with -t\n";

					}
					catch (format_Error & e)
					{
						cout << e.mesg << "\n";
					}
			}
			else

				if ((argvI == "-r") || (argvI == "-run"))
				{
					if (theNet == NULL)
						cout << "A network must be loaded before it is run.\n";
					else
						try
						{
							inputFile dataFile(argv[++i]);
							theNet->run(&dataFile, &callback_RunComplete);

							if (!quiet)
								cout << "Done with -r or -run\n";

						}
						catch (format_Error & e)
						{
							cout << e.mesg << "\n";
						}
				}
				else
					if (argvI == "-s")
					{
						if (theNet == NULL)
							cout << "A network must be loaded before it can be saved.\n";
						else
							try
							{
								theNet->saveTo(argv[++i]);

								if (!quiet)
									cout << "Done with -s\n";

							}
							catch (format_Error & e)
							{
								cout << e.mesg << "\n";
							}
					}
					else
						if (argvI == "-c")
						{
							if (theNet != NULL)
							{
								cout << "A network was already loaded and has been deleted.\n";
								delete theNet;
							}
							try
							{
								in = atoi(argv[++i]);
								out = atoi(argv[++i]);
								hidden = atoi(argv[++i]);
								fVal = 0.1;
								strArg = argv[++i];
								theNet = new nn(in, out, hidden, fVal, strArg);

								if (!quiet)
									cout << "Done with -c\n";

							}
							catch (format_Error & e)
							{
								cout << e.mesg << "\n";
							}
						}
						else
							if (argvI == "-rand")
							{
								if (theNet == NULL)
									cout << "A network must be loaded before it is randomised.\n";
								else
									try
									{
										theNet->randomise();

										if (!quiet)
											cout << "Done with -rand\n";

									}
									catch (format_Error & e)
									{
										cout << e.mesg << "\n";
									}

							}
							else
								if (argvI == "-at")
								{
									if (theNet == NULL)
										cout << "A network must be loaded before it is altered.\n";
									else
										try
										{
											vector<unsigned int> widths(3);

											in = atoi(argv[++i]);
											hidden = atoi(argv[++i]);
											out = atoi(argv[++i]);
											widths[0] = in;
											widths[1] = hidden;
											widths[2] = out;
											theNet->alter(&widths);

											if (!quiet)
												cout << "Done with -at\n";

										}
										catch (format_Error & e)
										{
											cout << e.mesg << "\n";
										}

								}
								else
									if (argvI == "-al")
									{
										if (theNet == NULL)
											cout << "A network must be loaded before it can be altered (-al).\n";
										else
											try
											{
												unsigned int layerNo;
												size_t colonPos;
												string modifier;
												string key;
												string value;


												layerNo = atoi(argv[++i]);
												modifier = argv[++i];
												colonPos = modifier.find(":");

												if (layerNo < 3) // need to check the layer number somewhere....
												{
													key = modifier.substr(0, colonPos);
													if (key == "biasNode")
													{
														if (layerNo < 2)		// maybe the layer check should be done by the network
														{
															value = modifier.substr(colonPos + 1);
															if (value == "true")
																theNet->alter(layerNo, BIAS_NODE, true);
															else
																if (modifier.substr(colonPos + 1, modifier.size() - colonPos) == "false")
																	theNet->alter(layerNo, BIAS_NODE, false);
																else
																	cout << "Invalid argument for modifier biasNode: " << value << "\n";
														}
														else
															cout << "Invalid layer for biasNode modifier: " << layerNo << "\n";
													}
													else
														if (key == "transition")
														{
															if (layerNo > 0)
															{
																value = modifier.substr(colonPos + 1);
																if (value == "SIGMOID")
																	theNet->alter(layerNo, TRANSITION_SIGMOID);
																else
																	if (value == "LINEAR")
																		theNet->alter(layerNo, TRANSITION_LINEAR);
																	else
																		if (value == "BINARY")
																			theNet->alter(layerNo, TRANSITION_BINARY);
																		else
																			cout << "Invalid argument for modifier biasNode: " << value << "\n";
															}
															else
																cout << "Invalid transition layer: " << layerNo << "\n";
														}
														else
															cout << "Modifer:" << key << " not valid\n";
												}
												else
													cout << "Invalid layer number:" << layerNo << "\n";


												if (!quiet)
													cout << "Done with -al\n";

											}
											catch (format_Error & e)
											{
												cout << e.mesg << "\n";
											}

									}
									else
										if (argvI == "-an")
										{
											if (theNet == NULL)
												cout << "A network must be loaded before it can be altered (-alt).\n";
											else
												try
												{
													unsigned int layerNo;
													unsigned int nodeNo;
													float newP;
													size_t colonPos;
													string modifier;
													string key;
													string value;


													layerNo = atoi(argv[++i]);
													nodeNo = atoi(argv[++i]);
													modifier = argv[++i];
													colonPos = modifier.find(":");

													if (layerNo == 0)
													{
														key = modifier.substr(0, colonPos);
														if (key == "p")
														{
															newP = (float)atof((modifier.substr(colonPos + 1)).c_str());
															theNet->alterNode(layerNo, nodeNo, newP);
														}
														else
															if (key == "pEqualsOneHalf")
															{
																value = modifier.substr(colonPos + 1);
																if (value == "true")
																	theNet->alterNode(layerNo, nodeNo, true);
																else
																	if (value == "false")
																		theNet->alterNode(layerNo, nodeNo, false);
																	else
																		cout << "Invalid argument for modifier pEqualsOneHalf: " << value << "\n";
															}
															else
																if (key == "input")
																{
																	value = modifier.substr(colonPos + 1);
																	if (value == "BINARY")
																		theNet->alterNode(layerNo, nodeNo, INPUT_BINARY);
																	else
																		if (value == "BIPOLAR")
																			theNet->alterNode(layerNo, nodeNo, INPUT_BIPOLAR);
																		else
																			if (value == "UNIFORM")
																				theNet->alterNode(layerNo, nodeNo, INPUT_UNIFORM);
																			else
																				cout << "Invalid argument for modifier input: " << value << "\n";

																}
																else
																	cout << "Modifer:" << key << " not valid\n";
													}
													else
														cout << "Invalid layer number:" << layerNo << "\n";


													if (!quiet)
														cout << "Done with -an\n";

												}
												catch (format_Error & e)
												{
													cout << e.mesg << "\n";
												}

										}
										else
											if (argvI == "-test")
											{
												if (theNet == NULL)
													cout << "A network must be loaded before it can be tested.\n";
												else
													try
													{
														trainingFile testFile(argv[++i]);
														theNet->test(&testFile, callback_TestComplete);

														if (!quiet)
															cout << "Done with -test\n";

													}
													catch (format_Error & e)
													{
														cout << e.mesg << "\n";
													}

											}
											else
												if (argvI == "-q+")
													quiet = true;
												else
													if (argvI == "-q-")
													{
														quiet = false;
														cout << "Done with -q-\n";
													}
													else
														if (argvI == "-v0")
														{
															if (theNet == NULL)
																cout << "A network must be loaded before its version can be reset.\n";
															else
																try
																{
																	theNet->resetVersion();

																	if (!quiet)
																		cout << "Done with -v\n";

																}
																catch (format_Error & e)
																{
																	cout << e.mesg << "\n";
																}

														}
														else
														{
															unknownFlag = true;
															cout << "Don't know that one:" << argvI << "\n";
														}

	}

	if (unknownFlag)
	{
		cout << "\n-n %file load a network from %file\n";
		cout << "-t %file training set %file and write the final error vector to standard output\n";
		cout << "(-r | -run) %file run from input set from inPath  and write each result vector to standard output\n";
		cout << "-rand randomise the network\n";
		cout << "-test %file run the input patter from training file %file and write the final difference vector to standard output\n";
		cout << "-s %path save on %path (with no trailing /) - use . to store in the current directory\n";
		cout << "-c %i %h %o %n create a new randomised network with %i input nodes %h hidden nodes %o output nodes, called %n (with a learning rate of 0.1)\n";
		cout << "-at %i %h %o alter the topology to be %i input nodes %h hidden nodes %o output nodes\n";
		cout << "-an %l %n %mod:%val alter the node %n on layer %l to have the modifer %mod with value %val\n";
		cout << "\tNode modifiers are:\n";
		cout << "\t\tp:%f set the p value to floating point number %f\n";
		cout << "\t\tpEqualsOneHalf:(true OR false) indicate that p is equal to exactly 0.5\n";
		cout << "\t\tinput:(UNIFORM OR BINARY OR BIPOLAR) indicate what type of input is expected by the node\n";
		cout << "-al %l mod:%value change the characteristics of layer %l\n";
		cout << "\tLayer modifiers are:\n";
		cout << "\t\tbiasNode:(true OR false) add or remove an input bias node to the layer %l\n";
		cout << "\t\ttransition:(SIGMOID OR LINEAR OR BINARY) change the transition function of layer %l\n";
		cout << "-q+ OR -q- Switch quiet mode on (-q+) or off (-q-) +q+ supresses the 'Done with...' after each command line arguement\n";
		cout << "-v0 reset the version of the network (major version == 0, minor verions == 0 and revision == 0)\n";
	}
	if (theNet != NULL)
		delete theNet;

	return 1;
}
