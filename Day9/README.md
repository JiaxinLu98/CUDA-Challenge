A convolution filter can be stored in constant memory because there are three properties:

1. The size of the convolution filter is small, and the radius of most convolution filters is 7 or smaller. --> The size of constant memory is also small.
2. The contents of the convolution filter do not change. -->  Values in constant memory cannot be modified by threads.
3. All threads access the convolution filter elements in the same order. --> The scope of constant memory is all thread blocks. 
