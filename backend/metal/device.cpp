#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "device.hpp"
#include <algorithm>
#include <fstream>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
namespace infeng::metal {
static std::string message(NS::Error* error) {
  auto* description = error ? error->localizedDescription() : nullptr;
  return description ? description->utf8String() : "Metal operation failed";
}
template <class T> T* require(T* value, const std::string& error) {
  if (!value)
    throw std::runtime_error(error);
  return value;
}
Device::Device(const std::filesystem::path& kernels) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  metalDevice = NS::TransferPtr(require(MTL::CreateSystemDefaultDevice(), "Metal device unavailable"));
  queue = NS::TransferPtr(require(metalDevice->newMTL4CommandQueue(), "MTL4 command queue unavailable"));
  allocator = NS::TransferPtr(require(metalDevice->newCommandAllocator(), "MTL4 command allocator unavailable"));
  auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
  descriptor->setInitialCapacity(512);
  NS::Error* error = nullptr;
  residency = NS::TransferPtr(require(metalDevice->newResidencySet(descriptor.get(), &error), message(error)));
  event = NS::TransferPtr(require(metalDevice->newSharedEvent(), "shared event creation failed"));
  queue->addResidencySet(residency.get());
  for (const auto& entry : std::filesystem::directory_iterator(kernels)) {
    if (entry.path().extension() != ".metal")
      continue;
    std::ifstream file(entry.path());
    std::stringstream stream;
    stream << file.rdbuf();
    auto source = NS::String::string(stream.str().c_str(), NS::UTF8StringEncoding);
    libraries.push_back(
        NS::TransferPtr(require(metalDevice->newLibrary(source, nullptr, &error), entry.path().filename().string() + ": " + message(error))));
  }
  constants = empty(1 << 18, true);
}
Device::~Device() {
  queue->removeResidencySet(residency.get());
}
void Device::add(MTL::Allocation* allocation) {
  residency->addAllocation(allocation);
  residency->commit();
}
void Device::remove(MTL::Allocation* allocation) {
  residency->removeAllocation(allocation);
  residency->commit();
}
void Device::wait() {
  uint64_t signal = ++eventValue;
  queue->signalEvent(event.get(), signal);
  event->waitUntilSignaledValue(signal, std::numeric_limits<uint64_t>::max());
}
Tensor Device::own(MTL::Buffer* buffer, uint64_t bytes) {
  add(buffer);
  return {std::shared_ptr<MTL::Buffer>(buffer,
                                       [this](MTL::Buffer* value) {
                                         remove(value);
                                         value->release();
                                       }),
          0, bytes};
}
Tensor Device::empty(uint64_t bytes, bool shared) {
  auto mode = shared ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate;
  return own(require(metalDevice->newBuffer(bytes, mode), "buffer allocation failed"), bytes);
}
Tensor Device::mapped(const std::filesystem::path& path) {
  uint64_t bytes = std::filesystem::file_size(path);
  uint64_t page = uint64_t(getpagesize());
  uint64_t mappedBytes = (bytes + page - 1) / page * page;
  int fd = open(path.c_str(), O_RDONLY);
  void* data = mmap(nullptr, mappedBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  close(fd);
  MTL::Buffer* buffer = metalDevice->newBuffer(data, mappedBytes, MTL::ResourceStorageModeShared, ^(void* pointer, NS::UInteger length) {
    munmap(pointer, length);
  });
  return own(buffer, bytes);
}
Pipeline* Device::pipeline(const std::string& name) {
  if (auto found = pipelines.find(name); found != pipelines.end())
    return found->second.get();
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  MTL::Function* function = nullptr;
  auto string = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
  for (const auto& library : libraries)
    if ((function = library->newFunction(string)))
      break;
  NS::Error* error = nullptr;
  Pipeline* pipelineState = require(metalDevice->newComputePipelineState(function, &error), name + ": " + message(error));
  pipelines.emplace(name, NS::TransferPtr(pipelineState));
  return pipelineState;
}
Device& Device::command() {
  constantOffset = tableIndex = 0;
  metalCommandBuffer = NS::TransferPtr(require(metalDevice->newCommandBuffer(), "MTL4 command buffer creation failed"));
  metalCommandBuffer->beginCommandBuffer(allocator.get());
  metalCommandBuffer->useResidencySet(residency.get());
  encoder = metalCommandBuffer->computeCommandEncoder();
  encoder->barrierAfterQueueStages(MTL::StageResourceState, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
  return *this;
}
MTL4::ArgumentTable* Device::table() {
  if (tableIndex < tables.size())
    return tables[tableIndex++].get();
  auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
  descriptor->setMaxBufferBindCount(24);
  descriptor->setInitializeBindings(true);
  NS::Error* error = nullptr;
  auto* argumentTable = require(metalDevice->newArgumentTable(descriptor.get(), &error), message(error));
  tables.push_back(NS::TransferPtr(argumentTable));
  ++tableIndex;
  return argumentTable;
}
void Device::commit() {
  encoder->endEncoding();
  metalCommandBuffer->endCommandBuffer();
  const MTL4::CommandBuffer* commands[] = {metalCommandBuffer.get()};
  queue->commit(commands, 1);
  wait();
  allocator->reset();
}
Tensor SparseKV::makeBuffer(uint32_t regions) {
  uint64_t bytes = regionBytes * regions;
  return device.own(
      require(device.metalDevice->newBuffer(bytes, MTL::ResourceStorageModePrivate, MTL::SparsePageSize256), "sparse KV allocation failed"), bytes);
}
SparseKV::SparseKV(Device& device, uint32_t virtualCount, uint32_t physicalCount, uint32_t layers, uint32_t drafterLayers)
    : device(device), virtualBlocks(virtualCount), maxPhysicalBlocks(physicalCount), layers(layers + drafterLayers), tilesPerBlock(this->layers * 2),
      blocksPerHeap(uint32_t(heapBytes / pageBytes) / tilesPerBlock), regionBytes(uint64_t(virtualBlocks) * pageBytes),
      resources{makeBuffer(this->layers), makeBuffer(this->layers)} {
}
void SparseKV::addHeap() {
  auto descriptor = NS::TransferPtr(MTL::HeapDescriptor::alloc()->init());
  descriptor->setType(MTL::HeapTypePlacement);
  descriptor->setStorageMode(MTL::StorageModePrivate);
  descriptor->setSize(heapBytes);
  descriptor->setSparsePageSize(MTL::SparsePageSize256);
  descriptor->setMaxCompatiblePlacementSparsePageSize(MTL::SparsePageSize256);
  MTL::Heap* heap = require(device.metalDevice->newHeap(descriptor.get()), "KV heap allocation failed");
  device.add(heap);
  heaps.push_back(NS::TransferPtr(heap));
  physicalBlocks += std::min(blocksPerHeap, maxPhysicalBlocks - physicalBlocks);
}
void SparseKV::ensure(uint32_t blocks) {
  while (physicalBlocks < blocks)
    addHeap();
}
void SparseKV::map(uint32_t virtualBlock, uint32_t physicalBlock) {
  MTL::Heap* heap = physicalBlock == UINT32_MAX ? nullptr : heaps[physicalBlock / blocksPerHeap].get();
  uint32_t heapOffset = physicalBlock == UINT32_MAX ? 0 : physicalBlock % blocksPerHeap * tilesPerBlock;
  auto mode = heap ? MTL::SparseTextureMappingModeMap : MTL::SparseTextureMappingModeUnmap;
  using Operation = MTL4::UpdateSparseBufferMappingOperation;
  for (uint32_t resource = 0; resource < 2; ++resource) {
    auto operation = [&](uint32_t layer) {
      return Operation{mode, NS::Range::Make(uint64_t(layer) * virtualBlocks + virtualBlock, 1), heap ? heapOffset + resource * layers + layer : 0};
    };
    Operation operations[]{operation(0), operation(1), operation(2), operation(3),  operation(4),  operation(5),  operation(6),
                           operation(7), operation(8), operation(9), operation(10), operation(11), operation(12), operation(13)};
    device.queue->updateBufferMappings(resources[resource].buffer.get(), heap, operations, layers);
  }
}
SparseKV::~SparseKV() {
  device.wait();
  for (Tensor& resource : resources)
    resource = {};
  for (const auto& heap : heaps)
    device.remove(heap.get());
}
} // namespace infeng::metal
