#include "array.h"
#include <iostream>
#include <cassert>
#include <execinfo.h>
#include <stdio.h>

namespace npycrf {
	namespace index {
		void check(int i, int size){
			if(0 <= i && i < size){
				return;
			}
			std::cout << i << " < " << size << std::endl;
			void* callstack[128];
			int frames = backtrace(callstack, 128);
			char** strs = backtrace_symbols(callstack, frames);
			for (int l = 0; l < frames; ++l) {
				printf("%s\n", strs[l]);
			}
			free(strs);
			assert(0 <= i && i < size);
		}
	}
}