#include <string>

template <typename T> inline const std::string typeName();

template <> const std::string typeName<char>() { return "int8"; }
template <> const std::string typeName<unsigned char>() { return "uint8"; }
template <> const std::string typeName<short>() { return "int16"; }
template <> const std::string typeName<unsigned short>() { return "uint16"; }
template <> const std::string typeName<int>() { return "int32"; }
template <> const std::string typeName<unsigned int>() { return "uint32"; }
template <> const std::string typeName<float>() { return "float32"; }
template <> const std::string typeName<long>() { return "int64"; }
template <> const std::string typeName<unsigned long>() { return "uint64"; }
template <> const std::string typeName<double>() { return "float64"; }