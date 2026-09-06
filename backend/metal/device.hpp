#pragma once
#include <Metal/Metal.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
namespace infeng::metal {
class Device;
struct Tensor {
  std::shared_ptr<MTL::Buffer> buffer;
  uint64_t offset = 0, bytes = 0;
  Tensor view(uint64_t byteOffset, uint64_t byteCount) const {
    return {buffer, offset + byteOffset, byteCount};
  }
  MTL::GPUAddress address() const {
    return buffer->gpuAddress() + offset;
  }
  template <class T> T* contents() const {
    return reinterpret_cast<T*>(static_cast<uint8_t*>(buffer->contents()) + offset);
  }
};
using Pipeline = MTL::ComputePipelineState;
class SparseKV;
class Device {
  friend class SparseKV;
  NS::SharedPtr<MTL::Device> metalDevice;
  NS::SharedPtr<MTL4::CommandQueue> queue;
  NS::SharedPtr<MTL4::CommandAllocator> allocator;
  NS::SharedPtr<MTL::ResidencySet> residency;
  NS::SharedPtr<MTL::SharedEvent> event;
  NS::SharedPtr<MTL4::CommandBuffer> metalCommandBuffer;
  MTL4::ComputeCommandEncoder* encoder = nullptr;
  Tensor constants;
  std::vector<NS::SharedPtr<MTL4::ArgumentTable>> tables;
  std::vector<NS::SharedPtr<MTL::Library>> libraries;
  std::unordered_map<std::string, NS::SharedPtr<Pipeline>> pipelines;
  uint64_t constantOffset = 0;
  uint64_t eventValue = 0;
  uint32_t tableIndex = 0;
  template <class T> void scalar(MTL4::ArgumentTable* table, uint32_t index, const T& value);
  MTL4::ArgumentTable* table();
  Tensor own(MTL::Buffer* buffer, uint64_t bytes);
  void add(MTL::Allocation* allocation);
  void remove(MTL::Allocation* allocation);
public:
  Device(const std::filesystem::path& kernels);
  Device(const Device&) = delete;
  ~Device();
  void wait();
  Tensor empty(uint64_t bytes, bool shared = false);
  Tensor mapped(const std::filesystem::path& path);
  Pipeline* pipeline(const std::string& name);
  Device& command();
  template <class... Scalars>
  void dispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors, const Scalars&... scalars);
  template <class... Scalars>
  void dispatch(const std::string& kernel, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors, const Scalars&... scalars);
  void copy(const Tensor& source, const Tensor& destination) {
    encoder->copyFromBuffer(source.buffer.get(), source.offset, destination.buffer.get(), destination.offset, destination.bytes);
  }
  void commit();
};
class SparseKV {
  static constexpr uint64_t pageBytes = 256ull << 10, heapBytes = 64ull << 20;
  Device& device;
  uint32_t virtualBlocks, maxPhysicalBlocks, layers, tilesPerBlock, physicalBlocks = 0, blocksPerHeap;
  uint64_t regionBytes;
  Tensor resources[2];
  std::vector<NS::SharedPtr<MTL::Heap>> heaps;
  Tensor makeBuffer(uint32_t regions);
  void addHeap();
  Tensor view(uint32_t resource, uint32_t region) const {
    return {resources[resource].buffer, uint64_t(region) * regionBytes, regionBytes};
  }
public:
  SparseKV(Device&, uint32_t virtualBlocks, uint32_t physicalBlocks, uint32_t targetLayers, uint32_t drafterLayers);
  ~SparseKV();
  void ensure(uint32_t blocks);
  void map(uint32_t virtualBlock, uint32_t physicalBlock);
  Tensor key(uint32_t layer) const {
    return view(0, layer);
  }
  Tensor value(uint32_t layer) const {
    return view(1, layer);
  }
  uint64_t mappedBytes() const {
    return uint64_t(physicalBlocks) * tilesPerBlock * pageBytes;
  }
};
template <class T> void Device::scalar(MTL4::ArgumentTable* table, uint32_t index, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  constantOffset = (constantOffset + 15) & ~15ull;
  std::memcpy(static_cast<uint8_t*>(constants.buffer->contents()) + constantOffset, &value, sizeof(T));
  table->setAddress(constants.address() + constantOffset, index);
  constantOffset += sizeof(T);
}
template <class... Scalars>
void Device::dispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors, const Scalars&... scalars) {
  auto* argumentTable = table();
  uint32_t index = 0;
  for (const Tensor& tensor : tensors)
    argumentTable->setAddress(tensor.address(), index++);
  (scalar(argumentTable, index++, scalars), ...);
  encoder->setComputePipelineState(pipeline);
  encoder->setArgumentTable(argumentTable);
  encoder->dispatchThreads(threads, group);
  encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}
template <class... Scalars>
void Device::dispatch(const std::string& kernel, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors,
                      const Scalars&... scalars) {
  dispatch(pipeline(kernel), threads, group, tensors, scalars...);
}
} // namespace infeng::metal
