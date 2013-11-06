
#include "nn.hpp"

void callback_RunComplete(const int index, void * caller)
{
	vector<float> * resVec;	// created by the net
	unsigned int i;

	resVec = ((nn*)caller)->runResult();
	cout << "Index: " << index << " results - ";
	for(i = 0; i< resVec->size(); i++)
		cout << i << ": " << (*resVec)[i];
	cout << "\n";
}

void callback_TrainingComplete(void * caller)
{
	vector<float> * errVec;	// created by the net
	unsigned int i;

	errVec = new vector<float>(((nn*)caller)->outputNodes());

	if ( ((nn*)caller)->trainingError(errVec) == SUCCESS)
	{
		cout << "Last Training Error Vector - ";
		for(i = 0; i< errVec->size(); i++)
			cout << i << ": " << (*errVec)[i];
		cout << "\n";
	}
	else
		cout << "Error: failed to retrieve the error vector.\n";
}

void callback_TestComplete(vector<float> * errVec, void * caller)
{
	unsigned int i;

	cout << "Test Results - ";
	for(i = 0; i< errVec->size(); i++)
		cout << i << ": " << (*errVec)[i];
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


	int i;

	string nnName = "TestNet";

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
					theNet = new nn(argv[++i]);

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
					cout << "A network must be loaded before it is trained.\n";
				else
					try
					{
						theNet->train(argv[++i], &callback_TrainingComplete);

						cout << "Done with -t\n";

					}
					catch (format_Error & e)
					{
						cout << e.mesg << "\n";
					}
			}
			else

				if ((argvI == "-r") || (argvI == "run"))
				{
					if (theNet == NULL)
						cout << "A network must be loaded before it is run.\n";
					else
						try
						{
							theNet->run(argv[++i], &callback_RunComplete);

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
							cout << "A network must be loaded before it is saved.\n";
						else
							try
							{
								theNet->saveTo(argv[++i]);

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
//								cout << in << " " << out << " " << hidden << " " << strArg << "\n";
								theNet = new nn(in, out, hidden, fVal, strArg);

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
											in = atoi(argv[++i]);
											hidden = atoi(argv[++i]);
											out = atoi(argv[++i]);
//											cout << in << " " << out << " " << hidden << " " << "\n";
											theNet->alter(in, hidden, out);

											cout << "Done with -at\n";

										}
										catch (format_Error & e)
										{
											cout << e.mesg << "\n";
										}

								}
								else
									if (argvI == "-am")
									{
										if (theNet == NULL)
											cout << "A network must be loaded before it is altered.\n";
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

//												cout << "-am, layer is:" << layerNo << " modifer is:" << modifier << "\n";

												if (layerNo == 0)
												{
													key = modifier.substr(0, colonPos);
													if (key == "biasNode")
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
														cout << "Modifer:" << key << " not valid\n";
												}
												else
													cout << "Invalid layer number:" << layerNo << "\n";


												cout << "Done with -am\n";

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
												cout << "A network must be loaded before it is tested.\n";
											else
												try
												{
//													cout << argv[++i] << " " << argv[++i] << " not specifically decoded\n";
													theNet->test(argv[++i], callback_TestComplete);

													cout << "Done with -test\n";

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
		cout << "(-r | -run) %file run from input set from inPath  and write each result vector to standard output\n";//and save the results in outPath\n";
		cout << "-rand randomise the network\n";
		cout << "-test %file run the input patter from training file %file and write the final difference vector to standard output\n";
		cout << "-s %path save on %path (with no trailing //\n";
		cout << "-c %i %h %o %n create a new randomised network with %i input nodes %h hidden nodes %o output nodes, called %n (with a learning rate of 0.1)\n";
		cout << "-at % %h %o alter the topology to be %i input nodes %h hidden nodes %o output nodes\n";
		cout << "-am 1 (biasNode:true OR biasNode:false) add or remove an input bias node to layer one\n";
	}
	if (theNet != NULL)
		delete theNet;

	return 1;
}
