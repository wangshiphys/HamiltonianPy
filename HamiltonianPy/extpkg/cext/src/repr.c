#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "bisearch.h"
#include "constant.h"
#include "numof1.h"

void hopping_C(const int cindex, const int aindex, const long *const base,
     const long dim, long *const row, long *const col, int *const elmts)
/*{{{*/
{
    long ket, judge, rule, i, j;
    int number;
    judge = (1<<cindex) + (1<<aindex);
    rule = (1<<aindex);

    if (cindex == aindex) {
        #pragma omp parallel if(dim>NUM_TO_PARALLEL) num_threads(NUM_THREADS)
        {
            //printf("The number of threads: %d\n", omp_get_num_threads());
            #pragma omp for private(i, ket)
            for (i=0; i<dim; ++i) {
                ket = *(base + i);
                if (ket & rule) {
                    *(row + i) = i;
                    *(col + i) = i;
                    *(elmts + i) = 1;
                }
            }
        }
    }
    else {
        #pragma omp parallel if(dim>NUM_TO_PARALLEL) num_threads(NUM_THREADS)
        {
            //printf("The number of threads: %d\n", omp_get_num_threads());
            #pragma omp for private(i, j, ket, number)
            for (i=0; i<dim; ++i) {
                ket = *(base + i);
                if ((ket & judge) == rule) {
                    ket ^= judge;
                    j = bisearch(ket, base, dim);
                    number = numof1(ket, cindex, aindex);
                    *(row + i) = j;
                    *(col + i) = i;
                    *(elmts + i) = 1 - 2 * (number & 1);
                }
            }
        }
    }
}
/*}}}*/

void hubbard_C(const int index0, const int index1, const long *const base,
     const long dim, long *const row, long *const col, int *const elmts)
/*{{{*/
{
    long ket, rule, i;
    if (index0 == index1) {
        rule = (1<<index0);
    }
    else {
        rule = (1<<index0) + (1<<index1);
    }

    #pragma omp parallel if(dim>NUM_TO_PARALLEL) num_threads(NUM_THREADS)
    {
        //printf("The number of threads: %d\n", omp_get_num_threads());
        #pragma omp for private(i, ket)
        for (i=0; i<dim; ++i) {
            ket = *(base + i);
            if ((ket & rule) == rule) {
                *(row + i) = i;
                *(col + i) = i;
                *(elmts + i) = 1;
            }
         
        }
    }
}
/*}}}*/

void pairing_C(const int index0, const int index1, const int otype,
     const long *const base, const long dim,
     long *const row, long *const col, int *const elmts)
/*{{{*/
{
    long ket, judge, rule, i, j;
    int number, revision;
    judge = (1<<index0) + (1<<index1);
    if (otype == CREATION) {
        rule = 0;
        if (index0 > index1) {
            revision = 1;
        }
        else {
            revision = 0;
        }
    }
    else if (otype == ANNIHILATION) {
        rule = judge;
        if (index0 > index1) {
            revision = 0;
        }
        else {
            revision = 1;
        }
    }
    else
    {
        fprintf(stderr, "Error in pairing, the wrong otype!\n");
        exit(EXIT_FAILURE);
    }

    if (index0 != index1) {
        #pragma omp parallel if(dim>NUM_TO_PARALLEL) num_threads(NUM_THREADS)
        {
            //printf("The number of threads: %d\n", omp_get_num_threads());
            #pragma omp for private(i, j, ket, number)
            for (i=0; i<dim;++i) {
                ket = *(base + i);
                if ((ket & judge) == rule) {
                    ket ^= judge;
                    j = bisearch(ket, base, dim);
                    number = numof1(ket, index0, index1) + revision;
                    *(row + i) = j;
                    *(col + i) = i;
                    *(elmts + i) = 1 - 2 * (number & 1);
                }
            }
        }
        
    }
}
/*}}}*/

void aoc_C(const int index, const int otype, const long *const lbase,
     const long *const rbase, const long ldim, const long rdim,
     long *const row, long *const col, int *const elmts)
/*{{{*/
{
    long ket, judge, rule, i, j;
    int number;
    judge = (1<<index);
    if (otype == CREATION) {
        rule = 0;
    }
    else if (otype == ANNIHILATION) {
        rule = judge;
    }
    else
    {
        fprintf(stderr, "Error in aoc, the wrong otype!\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel if(rdim>NUM_TO_PARALLEL) num_threads(NUM_THREADS)
    {
        //printf("The number of threads: %d\n", omp_get_num_threads());
        #pragma omp for private(i, j, ket, number)
        for (i=0; i<rdim; ++i) {
            ket = *(rbase + i);
            if ((ket & judge) == rule) {
                ket ^= judge;
                j = bisearch(ket, lbase, ldim);
                number = numof1(ket, -1, index);
                *(row + i) = j;
                *(col + i) = i;
                *(elmts + i) = 1 - 2 * (number & 1);
            }
        }
    }
}
/*}}}*/
