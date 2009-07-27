// This code is licensed under the:  Creative Commons - Attribution-Share 
// Alike 2.0 UK: England & Wales  License
// or the license certificate at:  http://creativecommons.org/licenses/by-sa/2.0/uk/
// contact Barrett@bv2.co.uk for other permissions

typedef unsigned char BYTE;
typedef unsigned short WORD;

WORD get_statusWord()
{
	
	WORD statusword;
	_asm
	{
		finit
		fstcw statusword
		fwait
	}
	return statusword;
}

void set_statusWord(WORD status_Word)
{
	_asm
	{
		fldcw status_Word   //has an fwait with it
	}
}


void set_Precision_Rounding(WORD precision,WORD rounding)
{
	WORD statusword;
	_asm 
	{
		push eax
		finit
		fstcw statusword
		fwait
		mov ax,statusword	  	  
		and ax,0f0ffh  
		or ax,rounding
		or ax,precision
		mov statusword,ax
		fldcw statusword
		pop eax
	}
}

#include <stdio.h>

void set_FPU_Precision_Rounding(BYTE precision,BYTE rounding)
{
	//precision can be 24, 53 ,64  ->  32, 64, 80  :  defaults to 80
	// rounding can be:
	//    0 : truncating mode
	//    1 : round to nearest or even      : default
	//    2 : round up
	//    3 : round down
	
	WORD precisionWord = 0;
	WORD roundingWord = 0;
	
	switch(precision) 
	{
        case 24:
            precisionWord = 0; 
			printf("Setting precision: %d\n", 32);
			
            break;
        case 53:
			precisionWord = 256;            
            break;
        default:
            precisionWord = 768;
            break;
    }
    
	switch(rounding) 
	{
        case 0:
            roundingWord = 3072; 
			printf("Setting rounding: %s\n", "round to nearest or even");

            break;
        case 2:
			roundingWord = 1024;            
            break;
        case 3:
			roundingWord = 2048;            
            break;
        default:
            roundingWord = 0;
            break;
    }
    
    set_Precision_Rounding(precisionWord,roundingWord);
}


