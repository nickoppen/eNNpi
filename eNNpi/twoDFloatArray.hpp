#ifndef _twoDFloatArray_h
#define _twoDFloatArray_h

#include <vector>
using namespace std ;

class twoDFloatArray
{
	// costruction destruction
	public:
                    twoDFloatArray(unsigned int d1, unsigned int d2)
                    {
                        dimension(d1, d2);
                    }

                    twoDFloatArray(unsigned int width)
                    {
                        dimension(1, width);
                    }

                    twoDFloatArray()
                    {
                        arr = NULL;
                    }

                    ~twoDFloatArray()
                    {
                        unsigned int i;
                        if (arr != NULL)
                        {
                            for(i = 0; i < arr->size(); i++)
                                delete arr->operator[](i);

                            delete arr;
                        }
                    }


    void 			dimension(unsigned int d1, unsigned int d2)
                    {
                        unsigned int i;

                        arr = new vector<vector<float>*> (d1);
                        for(i=0; i < arr->size(); i++)
                            arr->operator[](i) = new vector<float>(d2);
                    }

    void			redimension(unsigned int d1, unsigned int d2)
                    {
                        unsigned int i;

                        // clear out the old vectors first
                        if (arr != NULL)
                        {
                            for(i = 0; i < arr->size(); i++)
                                delete arr->operator[](i);

                            delete arr;
                        }

                        arr = new vector<vector<float>*> (d1);
                        for(i=0; i < arr->size(); i++)
                            arr->operator[](i) = new vector<float>(d2);
                    }

    void			addRow()
                    {
                        size_t arrWidth;


                        arrWidth = (arr->operator[](0))->size();
                        arr->push_back(new vector<float>(arrWidth));
                    }
	
	// access
    float 			value(unsigned int i, unsigned int j)
                    {
                        return (arr->operator[](i))->operator[](j);
                    }

    vector<float> * values(unsigned int i)
                    {
                        return arr->operator[](i);
                    }

    bool			dimensions(unsigned int &d1, unsigned int &d2)
                    {
                        // check that the array is dimensioned if not set return false

                        d2 = (arr->operator[](0))->size();
                        d1 = arr->size();

                        return true;
                    }

    float 			set(unsigned int i, unsigned int j, float f)
                    {
                        return (arr->operator[](i))->operator[](j) = f;
                    }

	
	// internals
	private:
					vector<vector<float> * > * arr;
	
};

#endif
