#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "device.hpp"
#include "model/qwen35.hpp"
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
Buffer::~Buffer() {
  device->remove(metalBuffer);
  metalBuffer->release();
}
Device::Device(const std::filesystem::path& kernels, bool profile) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  metalDevice = require(MTL::CreateSystemDefaultDevice(), "Metal device unavailable");
  if (!metalDevice->supportsPlacementSparse())
    throw std::runtime_error("placement sparse buffers unsupported");
  // create Metal 4 command submission, residency, and completion resources
  queue = require(metalDevice->newMTL4CommandQueue(), "MTL4 command queue unavailable");
  allocator = require(metalDevice->newCommandAllocator(), "MTL4 command allocator unavailable");
  auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
  descriptor->setInitialCapacity(512);
  NS::Error* error = nullptr;
  residency = require(metalDevice->newResidencySet(descriptor.get(), &error), message(error));
  event = require(metalDevice->newSharedEvent(), "shared event creation failed");
  queue->addResidencySet(residency);
  if (profile) {
    auto counters = NS::TransferPtr(MTL4::CounterHeapDescriptor::alloc()->init());
    counters->setType(MTL4::CounterHeapTypeTimestamp);
    counters->setCount(counterHeapEntries);
    counterHeap = require(metalDevice->newCounterHeap(counters.get(), &error), message(error));
    timestampScale = 1.0e9L / metalDevice->queryTimestampFrequency();
  }
  // find and compile all Metal sources into libraries
  std::vector<std::filesystem::path> sources;
  for (const auto& entry : std::filesystem::directory_iterator(kernels))
    if (entry.path().extension() == ".metal")
      sources.push_back(entry.path());
  std::sort(sources.begin(), sources.end());
  for (const auto& path : sources) {
    std::ifstream file(path);
    std::stringstream stream;
    stream << file.rdbuf();
    auto source = NS::String::string(stream.str().c_str(), NS::UTF8StringEncoding);
    libraries.push_back(require(metalDevice->newLibrary(source, nullptr, &error), path.filename().string() + ": " + message(error)));
  }
}
Device::~Device() {
  for (auto& [_, pipeline] : pipelines)
    pipeline->release();
  for (auto* library : libraries)
    library->release();
  if (counterHeap)
    counterHeap->release();
  event->release();
  queue->removeResidencySet(residency);
  residency->release();
  allocator->release();
  queue->release();
  metalDevice->release();
}
void Device::add(MTL::Allocation* allocation) {
  residency->addAllocation(allocation);
  residency->commit();
}
void Device::remove(MTL::Allocation* allocation) {
  residency->removeAllocation(allocation);
  residency->commit();
}
bool Device::wait() {
  uint64_t signal = ++eventValue;
  queue->signalEvent(event, signal);
  return event->waitUntilSignaledValue(signal, std::numeric_limits<uint64_t>::max());
}
void Device::recordKernel(const std::string& phase, const std::string& name, uint64_t gpuTimeNs) {
  auto found = std::find_if(kernelStats.begin(), kernelStats.end(),
                            [&](const KernelCounter& counter) { return counter.phase == phase && counter.name == name; });
  if (found == kernelStats.end())
    kernelStats.push_back({phase, name, gpuTimeNs, 1});
  else
    found->gpuTimeNs += gpuTimeNs, ++found->launches;
}
Tensor Device::empty(uint64_t bytes, bool shared) {
  MTL::Buffer* buffer =
      require(metalDevice->newBuffer(bytes, shared ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate), "buffer allocation failed");
  add(buffer);
  return {std::make_shared<Buffer>(this, buffer), 0, bytes};
}
Tensor Device::mapped(const std::filesystem::path& path) {
  uint64_t bytes = std::filesystem::file_size(path);
  uint64_t page = uint64_t(getpagesize());
  uint64_t mappedBytes = (bytes + page - 1) / page * page;
  int fd = open(path.c_str(), O_RDONLY);
  void* data = mmap(nullptr, mappedBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  close(fd);
  if (data == MAP_FAILED)
    throw std::runtime_error("failed to map " + path.filename().string());
  MTL::Buffer* buffer = metalDevice->newBuffer(data, mappedBytes, MTL::ResourceStorageModeShared, ^(void* pointer, NS::UInteger length) {
    munmap(pointer, length);
  });
  if (!buffer) {
    munmap(data, mappedBytes);
    throw std::runtime_error("failed to expose mapped weights to Metal");
  }
  add(buffer);
  return {std::make_shared<Buffer>(this, buffer), 0, bytes};
}
Tensor Device::upload(const void* source, uint64_t bytes) {
  Tensor target = empty(bytes);
  write(target, source, bytes);
  return target;
}
void Device::write(const Tensor& destination, const void* source, uint64_t bytes) {
  if (destination.buffer->metalBuffer->storageMode() == MTL::StorageModeShared) {
    std::memcpy(static_cast<uint8_t*>(destination.buffer->metalBuffer->contents()) + destination.offset, source, bytes);
    return;
  }
  // private buffers require a shared staging copy submitted before the host buffer is released
  MTL::Buffer* staging = require(metalDevice->newBuffer(source, bytes, MTL::ResourceStorageModeShared), "staging allocation failed");
  add(staging);
  CommandBuffer commands(*this, 0);
  commands.copy(staging, 0, destination.buffer->metalBuffer, destination.offset, bytes);
  commands.commit();
  remove(staging);
  staging->release();
}
Pipeline* Device::pipeline(const std::string& name) {
  if (auto found = pipelines.find(name); found != pipelines.end())
    return found->second;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  MTL::Function* function = nullptr;
  auto string = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
  // find the kernel function across loaded libraries and cache its pipeline state
  for (auto* library : libraries)
    if ((function = library->newFunction(string)))
      break;
  if (!function)
    throw std::runtime_error("missing kernel " + name);
  NS::Error* error = nullptr;
  Pipeline* pipelineState = require(metalDevice->newComputePipelineState(function, &error), name + ": " + message(error));
  function->release();
  pipelines.emplace(name, pipelineState);
  pipelineNames.emplace(pipelineState, name);
  return pipelineState;
}
CommandBuffer::CommandBuffer(Device& device, uint64_t constantBytes, bool profile, uint32_t dispatchCapacity, const char* passPhase)
    : device(device), constants(constantBytes ? device.empty(constantBytes, true) : Tensor{}), phase(passPhase) {
  metalCommandBuffer = require(device.metalDevice->newCommandBuffer(), "MTL4 command buffer creation failed");
  // begin recording, attach resource residency, and open the compute encoder
  metalCommandBuffer->beginCommandBuffer(device.allocator);
  metalCommandBuffer->useResidencySet(device.residency);

  if (profile && device.counterHeap) {
    counterHeap = device.counterHeap;
    counterLimit = 2 + 2 * dispatchCapacity;
    if (counterLimit > counterHeap->count())
      throw std::runtime_error("profiling forward exceeds the 4096-timestamp counter heap; use a shorter prefill");
    counterHeap->invalidateCounterRange(NS::Range::Make(0, counterLimit));
    counterIndex = 2;
    metalCommandBuffer->writeTimestampIntoHeap(counterHeap, 0);
  }
  encoder = metalCommandBuffer->computeCommandEncoder();
  encoder->barrierAfterQueueStages(MTL::StageResourceState, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}
MTL4::ArgumentTable* CommandBuffer::table(uint32_t index) {
  if (index < tables.size())
    return tables[index];
  auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
  descriptor->setMaxBufferBindCount(24);
  descriptor->setInitializeBindings(true);
  NS::Error* error = nullptr;
  auto* argumentTable = require(device.metalDevice->newArgumentTable(descriptor.get(), &error), message(error));
  tables.push_back(argumentTable);
  return argumentTable;
}
void CommandBuffer::copy(MTL::Buffer* source, uint64_t sourceOffset, MTL::Buffer* destination, uint64_t destinationOffset, uint64_t bytes) {
  encoder->copyFromBuffer(source, sourceOffset, destination, destinationOffset, bytes);
}
void CommandBuffer::submit() {
  // end, submit, wait for completion, resolve timestamps, and recycle command memory
  if (counterHeap)
    metalCommandBuffer->writeTimestampIntoHeap(counterHeap, 1);
  metalCommandBuffer->endCommandBuffer();
  const MTL4::CommandBuffer* commands[] = {metalCommandBuffer};
  device.queue->commit(commands, 1);
  if (!device.wait())
    throw std::runtime_error("Metal event wait timed out");
  if (counterHeap) {
    NS::Data* data = counterHeap->resolveCounterRange(NS::Range::Make(0, counterIndex));
    auto* timestamps = static_cast<const MTL4::TimestampHeapEntry*>(data->bytes());
    device.stats.gpuTimeNs += uint64_t((timestamps[1].timestamp - timestamps[0].timestamp) * device.timestampScale);
    ++device.stats.passes;
    for (const auto& profile : profiles)
      device.recordKernel(phase, profile.name,
                          uint64_t((timestamps[profile.end].timestamp - timestamps[profile.start].timestamp) * device.timestampScale));
  }
  device.allocator->reset();
}
CommandBuffer::~CommandBuffer() {
  for (auto* argumentTable : tables)
    argumentTable->release();
  metalCommandBuffer->release();
}
void CommandBuffer::commit() {
  encoder->endEncoding();
  submit();
}

Tensor SparseKV::makeBuffer(uint32_t regions) {
  uint64_t bytes = regionBytes * regions;
  MTL::Buffer* buffer =
      require(device.metalDevice->newBuffer(bytes, MTL::ResourceStorageModePrivate, MTL::SparsePageSize256), "sparse KV buffer allocation failed");
  device.add(buffer);
  return {std::make_shared<Buffer>(&device, buffer), 0, bytes};
}
SparseKV::SparseKV(Device& device, uint32_t virtualCount, uint32_t physicalCount, uint32_t layers, uint32_t drafterLayers)
    : device(device), virtualBlocks(virtualCount), maxPhysicalBlocks(physicalCount), tilesPerBlock((layers + drafterLayers) * 2),
      blocksPerHeap(uint32_t(heapBytes / pageBytes) / tilesPerBlock), regionBytes(uint64_t(virtualBlocks) * pageBytes),
      resources{{makeBuffer(layers), layers, 0},
                {makeBuffer(layers), layers, layers},
                {drafterLayers ? makeBuffer(drafterLayers == 1 ? 2 : drafterLayers) : Tensor{}, drafterLayers == 1 ? 2u : drafterLayers, layers * 2},
                {drafterLayers > 1 ? makeBuffer(drafterLayers) : Tensor{}, drafterLayers > 1 ? drafterLayers : 0, layers * 2 + drafterLayers}} {
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
  heaps.push_back(heap);
  physicalBlocks += std::min(blocksPerHeap, maxPhysicalBlocks - physicalBlocks);
}
void SparseKV::ensure(uint32_t blocks) {
  while (physicalBlocks < blocks)
    addHeap();
}
void SparseKV::update(const Resource& resource, uint32_t virtualBlock, MTL::Heap* heap, uint32_t heapOffset) {
  auto mode = heap ? MTL::SparseTextureMappingModeMap : MTL::SparseTextureMappingModeUnmap;
  using Operation = MTL4::UpdateSparseBufferMappingOperation;
  auto operation = [&](uint32_t region) {
    return Operation{mode, NS::Range::Make(uint64_t(region) * virtualBlocks + virtualBlock, 1), heap ? heapOffset + region : 0};
  };
  Operation operations[]{operation(0), operation(1), operation(2), operation(3), operation(4), operation(5), operation(6), operation(7)};
  device.queue->updateBufferMappings(resource.buffer.buffer->metalBuffer, heap, operations, resource.regions);
}
void SparseKV::map(uint32_t virtualBlock, uint32_t physicalBlock) {
  MTL::Heap* heap = heaps[physicalBlock / blocksPerHeap];
  uint32_t offset = physicalBlock % blocksPerHeap * tilesPerBlock;
  for (const Resource& resource : resources)
    if (resource.regions)
      update(resource, virtualBlock, heap, offset + resource.heapOffset);
}
void SparseKV::unmap(uint32_t virtualBlock) {
  for (const Resource& resource : resources)
    if (resource.regions)
      update(resource, virtualBlock, nullptr, 0);
}
SparseKV::~SparseKV() {
  device.wait();
  for (Resource& resource : resources)
    resource.buffer = {};
  for (auto* heap : heaps) {
    device.remove(heap);
    heap->release();
  }
}
} // namespace infeng::metal

namespace infeng::qwen35 {
namespace {
uint64_t hashBlock(uint64_t hash, const int32_t* tokens) {
  hash ^= 0x9e3779b97f4a7c15ull;
  for (uint32_t i = 0; i < blockTokens; ++i) {
    hash ^= uint32_t(tokens[i]) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    hash *= 0x100000001b3ull;
  }
  return hash;
}
void copyCheckpoint(Model& model, Sequence& sequence, const Tensor& arena, bool restore) {
  CommandBuffer copies(model.device, 0);
  for (uint32_t i = 0; i < model.layers.size(); ++i)
    if (!model.layers[i].fullAttention) {
      Tensor recurrent =
          model.states[i].recurrent[restore ? 0 : sequence.bank].view(uint64_t(sequence.slot) * recurrentStateBytes, recurrentStateBytes);
      Tensor conv = model.states[i].conv[restore ? 0 : sequence.bank].view(uint64_t(sequence.slot) * convStateBytes, convStateBytes);
      Tensor cachedRecurrent = arena.view(gdnOffset(i), recurrentStateBytes),
             cachedConv = arena.view(gdnOffset(i) + recurrentStateBytes, convStateBytes);
      copies.copy(restore ? cachedRecurrent : recurrent, restore ? recurrent : cachedRecurrent, recurrentStateBytes);
      copies.copy(restore ? cachedConv : conv, restore ? conv : cachedConv, convStateBytes);
    }
  if (model.hasMtp()) {
    Tensor seed = mtpSeed(model, sequence, restore ? 0 : sequence.bank), cached = arena.view(gdnCheckpointBytes, 8192);
    copies.copy(restore ? cached : seed, restore ? seed : cached, 8192);
  }
  copies.commit();
}
} // namespace

Tensor mtpSeed(Model& model, const Sequence& sequence, uint32_t bank) {
  return model.mtpSeeds.view(uint64_t(bank * maxBatchSequences + sequence.slot) * 8192, 8192);
}

uint8_t Model::acquireSlot() {
  auto slot = std::find(slots.begin(), slots.end(), false);
  if (slot == slots.end())
    throw std::runtime_error("maximum live sequence count reached");
  *slot = true;
  return slot - slots.begin();
}

void Model::bind(Sequence& sequence, uint32_t logical, uint32_t physical) {
  uint32_t block = physical == unbound ? sequence.bindings[logical] : physical;
  if (physical == unbound) {
    kv->unmap(uint32_t(sequence.slot) * maxLogicalBlocks + logical);
    --blocks[block].refs;
  } else {
    kv->map(uint32_t(sequence.slot) * maxLogicalBlocks + logical, physical);
    ++blocks[block].refs;
  }
  sequence.bindings[logical] = physical;
  blocks[block].touch = ++clock;
}

uint32_t Model::acquireBlock() {
  uint32_t physical = physicalBlocks < blocks.size() ? physicalBlocks++ : unbound;
  if (physical == unbound) {
    auto victim = std::min_element(blocks.begin(), blocks.end(), [](const PhysicalBlock& a, const PhysicalBlock& b) {
      return (a.refs ? UINT64_MAX : a.touch) < (b.refs ? UINT64_MAX : b.touch);
    });
    physical = victim - blocks.begin();
    auto found = prefixTable.find(blocks[physical].hash);
    if (found != prefixTable.end() && found->second == physical)
      prefixTable.erase(found);
  }
  blocks[physical] = {0, 0, ++clock};
  return physical;
}

void Model::reserve(const Batch& batch) {
  auto each = [&](auto action) {
    for (uint32_t row = 0; row < batch.size; ++row) {
      const Query& query = batch.queries[row];
      for (uint32_t logical = query.valid / blockTokens; logical <= (query.valid + query.count - 1) / blockTokens; ++logical)
        action(*query.sequence, logical);
    }
  };
  uint32_t required = 0;
  each([&](Sequence& sequence, uint32_t logical) { required += sequence.bindings[logical] == unbound; });
  uint32_t free = std::count_if(blocks.begin(), blocks.end(), [](const PhysicalBlock& block) { return !block.refs; });
  if (required > free)
    throw std::runtime_error("KV block pool has no evictable capacity");
  kv->ensure(physicalBlocks + std::min<uint32_t>(required, blocks.size() - physicalBlocks));
  each([&](Sequence& sequence, uint32_t logical) {
    if (sequence.bindings[logical] == unbound)
      bind(sequence, logical, acquireBlock());
    blocks[sequence.bindings[logical]].touch = ++clock;
  });
}

void Model::ensureCandidates(uint32_t rows) {
  if (rows <= candidateCapacity)
    return;
  for (uint32_t i = 0; i < layers.size(); ++i)
    if (!layers[i].fullAttention) {
      states[i].candidateConv = device.empty(convStateBytes * rows);
      states[i].candidateRecurrent = device.empty(recurrentStateBytes * rows);
    }
  candidateCapacity = rows;
}

uint32_t Model::lookupPrefix(Sequence& sequence, const int32_t* tokens, uint32_t length) {
  uint32_t limit = (length - 1) / blockTokens, checkpointBlock = unbound;
  uint64_t hash = 0, checkpointHash = 0;
  for (uint32_t logical = 0; logical < limit; ++logical) {
    hash = hashBlock(hash, tokens + uint64_t(logical) * blockTokens);
    auto prefix = prefixTable.find(hash);
    if (prefix == prefixTable.end())
      break;
    if (!((logical + 1) * blockTokens % gdnCheckpointTokens) && checkpointCache.count(hash)) {
      checkpointBlock = logical;
      checkpointHash = hash;
    }
  }
  if (checkpointBlock == unbound)
    return 0;
  HybridCheckpoint& checkpoint = checkpointCache.at(checkpointHash);
  checkpoint.touch = ++clock;
  copyCheckpoint(*this, sequence, checkpoint.arena, true);
  hash = 0;
  for (uint32_t logical = 0; logical <= checkpointBlock; ++logical) {
    hash = hashBlock(hash, tokens + uint64_t(logical) * blockTokens);
    bind(sequence, logical, prefixTable.at(hash));
  }
  sequence.bank = 0;
  return (checkpointBlock + 1) * blockTokens;
}

void Model::publishPrefix(Sequence& sequence, const int32_t* tokens, uint32_t tokenOffset, uint32_t oldValid, uint32_t valid) {
  try {
    uint32_t first = oldValid / blockTokens, count = valid / blockTokens;
    if (first >= count)
      return;
    uint64_t hash = first ? blocks[sequence.bindings[first - 1]].hash : 0;
    for (uint32_t logical = first; logical < count; ++logical) {
      hash = hashBlock(hash, tokens + uint64_t(logical) * blockTokens - tokenOffset);
      uint32_t physical = sequence.bindings[logical];
      blocks[physical].hash = hash;
      prefixTable.emplace(hash, physical);
      blocks[physical].touch = ++clock;
    }
    if (valid % gdnCheckpointTokens)
      return;
    hash = blocks[sequence.bindings[valid / blockTokens - 1]].hash;
    if (checkpointCache.count(hash)) {
      checkpointCache.at(hash).touch = ++clock;
      return;
    }
    if (checkpointCache.size() == maxGdnCheckpoints) {
      auto victim = std::min_element(checkpointCache.begin(), checkpointCache.end(),
                                     [](const auto& a, const auto& b) { return a.second.touch < b.second.touch; });
      checkpointCache.erase(victim);
    }
    HybridCheckpoint checkpoint{device.empty(gdnCheckpointBytes + (hasMtp() ? 8192 : 0)), ++clock};
    copyCheckpoint(*this, sequence, checkpoint.arena, false);
    checkpointCache.emplace(hash, std::move(checkpoint));
  } catch (...) {
  }
}

void Model::release(Sequence& sequence) {
  for (uint32_t logical = 0; logical < sequence.bindings.size(); ++logical)
    if (sequence.bindings[logical] != unbound)
      bind(sequence, logical, unbound);
  slots[sequence.slot] = false;
}

Sequence::Sequence(Model& owner) : model(owner), bindings(owner.maxLogicalBlocks, unbound), slot(model.acquireSlot()) {
}

Sequence::~Sequence() {
  model.release(*this);
}
} // namespace infeng::qwen35
