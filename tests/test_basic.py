from pyrvl import compress, decompress
import numpy as np

def test_compress():
    for i in range(10):
        width= np.random.randint(0, 8192)
        height= np.random.randint(0, 8192)
        input = np.random.randint(0, 65535, (height, width), dtype=np.uint16)
        compressed = compress(input)
        decompressed = decompress(compressed)
        assert np.array_equal(input, decompressed)