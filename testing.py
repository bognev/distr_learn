import timeit

py = timeit.timeit('''example_original.test(5000)''', setup='import example_original', number=1000)
cy = timeit.timeit('''example_cython.test(5000)''', setup='import example_cython', number=1000)

print(py, cy)
print(py/cy)