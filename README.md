# sparseca

Implements correspondence analysis algorithms as described in:
        Greenacre, Michael. Correspondence Analysis in Practice. 2007. 2nd 
            Edition. Boca Raton, Fl.: Chapman and Hall/CRC Press.
    
    The innovation in this implementation is to use Truncated SVD and a few 
    other tricks to reduce the memory complexity of the algorithm. As a result, 
    the algorithm is tractable for larger matrices that would defeat existing 
    implementations on most desktop machines.
    
    This implementation calculates the total inertia of the correspondence 
    matrix by computing the trace of the squared correspondence matrix (C'C), 
    and it computes the trace by summing over the dot products of each column 
    vector with itself. As a result, this implementation is most efficient
    when N >> M, that is, when the number of rows is much greater than the 
    number of columns. 
    
    That final step is the only one currently parallelized using multiprocessing
    (if cores > 1).

    This was developed for use in a specific application (my dissertation) and aspects of correspondence analysis I didn't need or use are not fully implemented.
