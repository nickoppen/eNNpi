
#include "nn.hpp"

void callback_RunComplete(const int index, void * caller)
{
	vector<float> * resVec;	// created by the net
	unsigned int i;

	resVec = ((nn*)caller)->runResult();
	for(i = 0; i< resVec->size(); i++)
		cout << i << ": " << (*resVec)[i];
	cout << "\nIndex: " << index << " is complete.\n";
}

void callback_TrainingComplete(void * caller)
{
	vector<float> * errVec;	// created by the net
	unsigned int i;

	errVec = ((nn*)caller)->trainingError();
	for(i = 0; i< errVec->size(); i++)
		cout << i << ": " << (*errVec)[i];
	cout << "\nDone with Training\n";
}

int main(int argc, char *argv[])
{
	nn * theNet = NULL;
//	networkFile * nFile = NULL;
//	trainingFile * trFile = NULL;
//	inputFile * inFile = NULL;
//	fstream * rawFile = NULL;
	string argvI;
	string strArg;
	string strSave;
	int in, out, hidden;
	float fVal;
	bool unknownFlag = false;


	int i;

	/*
	 * -n %path network Path
	 * -t %path training set path
	 * -r %path run from input set in path
	 * -s %path save on path
	 * -c %i %h %o %n create a new randomised network with %i input nodes %h hidden nodes %o output nodes, called %n
	 */
	string nnName = "TestNet";

	for (i=1; i<argc; i++)
	{
		argvI = argv[i];
		if (argvI == "-n")
		{
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
					cout << "A network must be loaded before it is trained";
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

				if (argvI == "-r")
				{
					if (theNet == NULL)
						cout << "A network must be loaded before it is run";
					else
						try
						{
							theNet->run(argv[++i], &callback_RunComplete);

							cout << "Done with -r\n";

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
							cout << "A network must be loaded before it is saved";
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
							if (theNet != NULL)		// delete the old network if there is one
							{
								cout << "the net is not null";
								delete theNet;
							}
							try
							{
								in = atoi(argv[++i]);
								out = atoi(argv[++i]);
								hidden = atoi(argv[++i]);
								fVal = 0.1;
								strArg = argv[++i];
								cout << in << " " << out << " " << hidden << " " << strArg << "\n";
								theNet = new nn(in, out, hidden, fVal, strArg);
								theNet->randomise();

								cout << "Done with -c\n";

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
		cout << "\n-n %path load a network from %path\n";
		cout << "-t %path training set %path\n";
		cout << "-r %inPath %outPath run from input set from inPath and save the results in outPath\n";
		cout << "-s %path save on %path\n";
		cout << "-c %i %h %o %n create a new randomised network with %i input nodes %h hidden nodes %o output nodes, called %n\n";
	}
	if (theNet != NULL)
		delete theNet;

	return 1;
}
