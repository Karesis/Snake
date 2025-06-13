#include <stdlib.h>
#include <stdio.h>
#include "_malloc.h"

void* 
_safe_malloc_internal(size_t size, const char* file, int line) 
{
    if (size == 0)
    {
        fprintf(stderr, "WARN: malloc(0 bytes) called at %s:%d\n", file, line);
        return NULL;
    }
    void* ptr = malloc(size);
    if (ptr == NULL)
    {
        perror(NULL);
        fprintf(stderr, "FATAL: malloc(%zu bytes) failed at %s:%d\n", size, file, line);
        return NULL;
    }
    return ptr;
}

void* 
_safe_calloc_internal(size_t num, size_t size, const char* file, int line) 
{
    if (num == 0 || size == 0) 
    {
        fprintf(stderr, "WARN: calloc(num=0 or size=0) called at %s:%d\n", file, line);
        return NULL;
    }
    void* ptr = calloc(num, size);
    if (ptr == NULL)
    {
        perror(NULL);
        fprintf(stderr, "FATAL: calloc(%zu, %zu bytes) failed at %s:%d\n", num, size, file, line);
        return NULL;
    }
    return ptr;
}
