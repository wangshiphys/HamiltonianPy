/*Implement the binary search algorithm!*/

#include <stdio.h>
#include <stdlib.h>

long bisearch(const long aim, const long *list, const long n)
{
    long low, high, mid, buff;
    low = 0;
    high = n - 1;
    while (low <= high) {
        mid = (low + high) / 2;
        buff = *(list + mid);
        if (aim==buff) {
            return mid;
        }
        else if (aim>buff) {
            low = mid + 1;
        }
        else {
            high = mid - 1;
        }
    }
    fprintf(stderr, "Error in bisearch function, "
            "the aim does not contained in the array!\n");
    exit(EXIT_FAILURE);
}
