#ifndef _MALLOC_H
#define _MALLOC_H

#include <stdlib.h> 

void* _safe_malloc_internal(size_t size, const char* file, int line);
#define safemalloc(size) _safe_malloc_internal(size, __FILE__, __LINE__)

void* _safe_calloc_internal(size_t num, size_t size, const char* file, int line);
#define safecalloc(num, size) _safe_calloc_internal(num, size, __FILE__, __LINE__)

#endif // _MALLOC_H

