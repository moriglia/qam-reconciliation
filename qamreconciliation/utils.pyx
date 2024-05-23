


cpdef double dist_cut(double x):
    if x < 0:
        return 0
    if x > 1:
        return 1
    return x



cpdef int count_errors_from_lappr(double [:] lappr, unsigned char [:] word):
    cdef int count = 0
    cdef int i

    if (lappr.size != word.size):
        raise ValueError(f"Sizes do not match")

    for i in range(lappr.size):
        if (lappr[i] >= 0):
            count += word[i]
        else:
            count += 1-word[i]

    return count
