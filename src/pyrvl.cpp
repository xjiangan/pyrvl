#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;
using namespace nb::literals;

class RvlCodec
{
public:
  RvlCodec();
  // Compress input data into output. The size of output can be equal to
  // (1.5 * numPixels + 4) in the worst case.
  int CompressRVL(
      const uint16_t *input, unsigned char *output,
      int numPixels);
  // Decompress input data into output. The size of output must be
  // equal to numPixels.
  void DecompressRVL(
      const unsigned char *input, uint16_t *output,
      int numPixels);

private:
  RvlCodec(const RvlCodec &);
  RvlCodec &operator=(const RvlCodec &);

  void EncodeVLE(int value);
  int DecodeVLE();

  int *buffer_;
  int *pBuffer_;
  int word_;
  int nibblesWritten_;
};

RvlCodec::RvlCodec() {}

void RvlCodec::EncodeVLE(int value)
{
  do
  {
    int nibble = value & 0x7; // lower 3 bits
    if (value >>= 3)
    {
      nibble |= 0x8;
    } // more to come
    word_ <<= 4;
    word_ |= nibble;
    if (++nibblesWritten_ == 8)
    { // output word
      *pBuffer_++ = word_;
      nibblesWritten_ = 0;
      word_ = 0;
    }
  } while (value);
}

int RvlCodec::DecodeVLE()
{
  unsigned int nibble;
  int value = 0, bits = 29;
  do
  {
    if (!nibblesWritten_)
    {
      word_ = *pBuffer_++; // load word
      nibblesWritten_ = 8;
    }
    nibble = word_ & 0xf0000000;
    value |= (nibble << 1) >> bits;
    word_ <<= 4;
    nibblesWritten_--;
    bits -= 3;
  } while (nibble & 0x80000000);
  return value;
}

int RvlCodec::CompressRVL(
    const uint16_t *input, unsigned char *output,
    int numPixels)
{
  buffer_ = pBuffer_ = reinterpret_cast<int *>(output);
  nibblesWritten_ = 0;
  const uint16_t *end = input + numPixels;
  uint16_t previous = 0;
  while (input != end)
  {
    int zeros = 0, nonzeros = 0;
    for (; (input != end) && !*input; input++, zeros++)
    {
    }
    EncodeVLE(zeros); // number of zeros
    for (const uint16_t *p = input; (p != end) && *p++; nonzeros++)
    {
    }
    EncodeVLE(nonzeros); // number of nonzeros
    for (int i = 0; i < nonzeros; i++)
    {
      uint16_t current = *input++;
      int delta = current - previous;
      int positive = (delta << 1) ^ (delta >> 31);
      EncodeVLE(positive); // nonzero value
      previous = current;
    }
  }
  if (nibblesWritten_)
  { // last few values
    *pBuffer_++ = word_ << 4 * (8 - nibblesWritten_);
  }
  return static_cast<int>((unsigned char *)pBuffer_ - (unsigned char *)buffer_); // num bytes
}

void RvlCodec::DecompressRVL(
    const unsigned char *input, uint16_t *output,
    int numPixels)
{
  buffer_ = pBuffer_ = const_cast<int *>(reinterpret_cast<const int *>(input));
  nibblesWritten_ = 0;
  uint16_t current, previous = 0;
  int numPixelsToDecode = numPixels;
  while (numPixelsToDecode)
  {
    int zeros = DecodeVLE(); // number of zeros
    numPixelsToDecode -= zeros;
    for (; zeros; zeros--)
    {
      *output++ = 0;
    }
    int nonzeros = DecodeVLE(); // number of nonzeros
    numPixelsToDecode -= nonzeros;
    for (; nonzeros; nonzeros--)
    {
      int positive = DecodeVLE(); // nonzero value
      int delta = (positive >> 1) ^ -(positive & 1);
      current = previous + delta;
      *output++ = current;
      previous = current;
    }
  }
}

NB_MODULE(pyrvl, m)
{
  m.doc() = "RVL compression and decompression module";
  m.def("compress", [](const nb::ndarray<uint16_t, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig>& arr)
        {
        RvlCodec rvl;
        uint16_t h = arr.shape(0),w = arr.shape(1);
        int numPixels = h * w;
        unsigned char * buffer = new unsigned char[(int)(3 * numPixels + 8+4)];
        uint16_t * shape = (uint16_t *)buffer;
        shape[0] = h;
        shape[1] = w;
        int size = rvl.CompressRVL(arr.data(), buffer+4, numPixels);
        nb::bytes output(buffer, size+4); // PyBytes_FromStringAndSize: new bytes object with a copy
        delete[] buffer;
        return output; }, "Compress a numpy array using RVL compression");
  m.def("decompress", [](nb::bytes input)
        {
        RvlCodec rvl;
        const unsigned char * inputData = reinterpret_cast<const unsigned char *>(input.data());
        uint16_t * shape = (uint16_t *)inputData;
        int h = shape[0];
        int w = shape[1];
        int numPixels = h * w;
        inputData += 4;
        uint16_t * output = new uint16_t[numPixels];
        rvl.DecompressRVL(inputData, output, numPixels);
        nb::capsule owner(output, [](void *p) noexcept {
             delete[] (uint16_t *) p;
          });
        nb::ndarray<nb::numpy, uint16_t, nb::ndim<2>> arr(output, {h,w},owner);
        return arr;}, "Decompress a RVL compressed numpy array");
}
