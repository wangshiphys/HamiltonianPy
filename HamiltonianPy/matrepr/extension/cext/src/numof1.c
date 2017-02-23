/*Count the number of 1 between position p0 and p1 */
/*in the binary representation of integer: num.*/

int numof1(long num, int p0, int p1)
{
    int res, swap, i;
    res = 0;

    if (p0 > p1) {
        swap = p1;
        p1 = p0;
        p0 = swap;
    }

    if ((p1 - p0) > 1) {
        num >>= (p0+1);
        for (i=0; i<(p1 - p0 - 1); ++i) {
            res += (num >> i) & 1;
        }
    }

    return res;
}
