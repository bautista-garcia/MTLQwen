#pragma once
#include <Metal/Metal.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>
namespace infeng::metal {
struct Counters {
  uint64_t gpuTimeNs = 0, passes = 0;
};
struct KernelCounter {
  std::string phase, name;
  uint64_t gpuTimeNs = 0, launches = 0;
};
class Device;
struct Buffer {
  Device* device;
  MTL::Buffer* metalBuffer;
  Buffer(Device* device, MTL::Buffer* metalBuffer) : device(device), metalBuffer(metalBuffer) {
  }
  ~Buffer();
};
struct Tensor {
  std::shared_ptr<Buffer> buffer;
  uint64_t offset = 0, bytes = 0;
  Tensor view(uint64_t byteOffset, uint64_t byteCount) const {
    return {buffer, offset + byteOffset, byteCount};
  }
  MTL::GPUAddress address() const {
    return buffer->metalBuffer->gpuAddress() + offset;
  }
  template <class T> T* contents() const {
    return reinterpret_cast<T*>(static_cast<uint8_t*>(buffer->metalBuffer->contents()) + offset);
  }
};
using Pipeline = MTL::ComputePipelineState;
class CommandBuffer {
  friend class Device;
  Device& device;
  MTL4::CommandBuffer* metalCommandBuffer = nullptr;
  MTL4::ComputeCommandEncoder* encoder = nullptr;
  MTL4::CounterHeap* counterHeap = nullptr;
  Tensor constants;
  std::vector<MTL4::ArgumentTable*> tables;
  struct KernelSample {
    std::string name;
    uint32_t start, end;
  };
  std::vector<KernelSample> profiles;
  std::string phase;
  uint64_t constantOffset = 0;
  uint32_t tableIndex = 0, counterIndex = 0, counterLimit = 0;
  template <class T> void scalar(MTL4::ArgumentTable* table, uint32_t index, const T& value);
  template <class... Scalars>
  void encodeDispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors, const Scalars&... scalars);
  MTL4::ArgumentTable* table(uint32_t index);
  void copy(MTL::Buffer* source, uint64_t sourceOffset, MTL::Buffer* destination, uint64_t destinationOffset, uint64_t bytes);
  void submit();

public:
  explicit CommandBuffer(Device& device, uint64_t constantBytes = 1 << 20, bool profile = false, uint32_t dispatchCapacity = 1024,
                         const char* phase = "unknown");
  CommandBuffer(const CommandBuffer&) = delete;
  ~CommandBuffer();
  template <class... Scalars>
  void dispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors, const Scalars&... scalars);
  void copy(const Tensor& source, const Tensor& destination, uint64_t bytes) {
    copy(source.buffer->metalBuffer, source.offset, destination.buffer->metalBuffer, destination.offset, bytes);
  }
  void commit();
};
class Device {
  friend struct Buffer;
  friend class CommandBuffer;
  friend class SparseKV;
  static constexpr uint32_t counterHeapEntries = 4096;
  MTL::Device* metalDevice = nullptr;
  MTL4::CommandQueue* queue = nullptr;
  MTL4::CommandAllocator* allocator = nullptr;
  MTL::ResidencySet* residency = nullptr;
  MTL::SharedEvent* event = nullptr;
  MTL4::CounterHeap* counterHeap = nullptr;
  std::vector<MTL::Library*> libraries;
  std::unordered_map<std::string, Pipeline*> pipelines;
  std::unordered_map<Pipeline*, std::string> pipelineNames;
  std::vector<KernelCounter> kernelStats;
  uint64_t eventValue = 0;
  long double timestampScale = 0;
  void add(MTL::Allocation* allocation);
  void remove(MTL::Allocation* allocation);
  bool wait();
  void recordKernel(const std::string& phase, const std::string& name, uint64_t gpuTimeNs);

public:
  Device(const std::filesystem::path& kernels, bool profile);
  Device(const Device&) = delete;
  ~Device();
  Tensor empty(uint64_t bytes, bool shared = false);
  Tensor mapped(const std::filesystem::path& path);
  Tensor upload(const void* source, uint64_t bytes);
  void write(const Tensor& destination, const void* source, uint64_t bytes);
  Pipeline* pipeline(const std::string& name);
  const std::vector<KernelCounter>& kernelCounters() const {
    return kernelStats;
  }
  Counters stats;
};
class SparseKV {
  static constexpr uint64_t pageBytes = 256ull << 10, heapBytes = 64ull << 20;
  struct Resource {
    Tensor buffer;
    uint32_t regions, heapOffset;
  };
  Device& device;
  uint32_t virtualBlocks, maxPhysicalBlocks, tilesPerBlock, physicalBlocks = 0, blocksPerHeap;
  uint64_t regionBytes;
  Resource resources[4];
  std::vector<MTL::Heap*> heaps;
  Tensor makeBuffer(uint32_t regions);
  void addHeap();
  void update(const Resource&, uint32_t virtualBlock, MTL::Heap*, uint32_t heapOffset);
  Tensor view(uint32_t resource, uint32_t region) const {
    return {resources[resource].buffer.buffer, uint64_t(region) * regionBytes, regionBytes};
  }

public:
  SparseKV(Device&, uint32_t virtualBlocks, uint32_t physicalBlocks, uint32_t targetLayers, uint32_t drafterLayers);
  ~SparseKV();
  void ensure(uint32_t blocks);
  void map(uint32_t virtualBlock, uint32_t physicalBlock);
  void unmap(uint32_t virtualBlock);
  Tensor key(uint32_t layer) const {
    return layer < resources[0].regions ? view(0, layer) : view(2, layer - resources[0].regions);
  }
  Tensor value(uint32_t layer) const {
    if (layer < resources[1].regions)
      return view(1, layer);
    uint32_t drafter = layer - resources[1].regions;
    return resources[3].regions ? view(3, drafter) : view(2, resources[2].regions / 2 + drafter);
  }
  uint64_t mappedBytes() const {
    return uint64_t(physicalBlocks) * tilesPerBlock * pageBytes;
  }
};
template <class T> void CommandBuffer::scalar(MTL4::ArgumentTable* table, uint32_t index, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  constantOffset = (constantOffset + 15) & ~15ull;
  if (constantOffset + sizeof(T) > constants.bytes)
    throw std::runtime_error("command buffer constant storage exceeded");
  std::memcpy(static_cast<uint8_t*>(constants.buffer->metalBuffer->contents()) + constantOffset, &value, sizeof(T));
  table->setAddress(constants.address() + constantOffset, index);
  constantOffset += sizeof(T);
}
template <class... Scalars>
void CommandBuffer::encodeDispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors,
                                   const Scalars&... scalars) {
  uint32_t start = 0, end = 0;
  if (counterHeap) {
    if (counterIndex + 2 > counterLimit)
      throw std::runtime_error("command buffer counter storage exceeded");
    start = counterIndex++;
    encoder->writeTimestamp(MTL4::TimestampGranularityPrecise, counterHeap, start);
  }
  auto* argumentTable = table(tableIndex++);
  uint32_t index = 0;
  for (const Tensor& tensor : tensors)
    argumentTable->setAddress(tensor.address(), index++);
  (scalar(argumentTable, index++, scalars), ...);
  encoder->setComputePipelineState(pipeline);
  encoder->setArgumentTable(argumentTable);
  encoder->dispatchThreads(threads, group);
  encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
  if (counterHeap) {
    end = counterIndex++;
    encoder->writeTimestamp(MTL4::TimestampGranularityPrecise, counterHeap, end);
    profiles.push_back({device.pipelineNames.at(pipeline), start, end});
  }
}
template <class... Scalars>
void CommandBuffer::dispatch(Pipeline* pipeline, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors,
                             const Scalars&... scalars) {
  encodeDispatch(pipeline, threads, group, tensors, scalars...);
}
} // namespace infeng::metal
