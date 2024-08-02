cimport cython


cpdef double dist_cut(double x):
    if x < 0:
        return 0
    if x > 1:
        return 1
    return x



cpdef int count_errors_from_lappr(double [:] lappr, unsigned char [:] word) noexcept nogil:
    cdef int count = 0
    cdef int i, s_lappr, s_word

    s_lappr = len(lappr)
    s_word = len(word)

    if (s_lappr != s_word):
        return max(s_lappr, s_word)

    for i in range(s_lappr):
        with cython.boundscheck(False), cython.wraparound(False):
            if (lappr[i] >= 0):
                count += word[i]
            else:
                count += 1-word[i]

    return count
