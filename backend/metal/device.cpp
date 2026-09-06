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
      return Operation{mode, NS::Range::Make(uint64_t(layer) * virtualBlocks + virtualBlock, 1),
                       heap ? heapOffset + resource * layers + layer : 0};
    };
    Operation operations[]{operation(0), operation(1), operation(2),  operation(3),  operation(4),  operation(5),  operation(6),
                           operation(7), operation(8), operation(9),  operation(10), operation(11), operation(12), operation(13)};
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
void copyCheckpoint(Engine& model, Sequence& sequence, const Tensor& arena, bool restore) {
  Device& copies = model.device.command();
  for (uint32_t i = 0; i < model.layers.size(); ++i)
    if ((i + 1) % fullAttentionInterval) {
      GdnState state = gdnState(model.gdnStates[restore ? 0 : sequence.bank], maxBatchSequences, i);
      Tensor recurrent = state.recurrent.view(uint64_t(sequence.slot) * recurrentStateBytes, recurrentStateBytes);
      Tensor conv = state.conv.view(uint64_t(sequence.slot) * convStateBytes, convStateBytes);
      Tensor cachedRecurrent = arena.view(gdnOffset(i), recurrentStateBytes),
             cachedConv = arena.view(gdnOffset(i) + recurrentStateBytes, convStateBytes);
      copies.copy(restore ? cachedRecurrent : recurrent, restore ? recurrent : cachedRecurrent);
      copies.copy(restore ? cachedConv : conv, restore ? conv : cachedConv);
    }
  if (model.drafter == Drafter::mtp) {
    Tensor seed = mtpSeed(model, sequence, restore ? 0 : sequence.bank), cached = arena.view(gdnCheckpointBytes, 8192);
    copies.copy(restore ? cached : seed, restore ? seed : cached);
  }
  copies.commit();
}
} // namespace
Tensor mtpSeed(Engine& model, const Sequence& sequence, uint32_t bank) {
  return model.mtpSeeds.view(uint64_t(bank * maxBatchSequences + sequence.slot) * 8192, 8192);
}
void Engine::bind(Sequence& sequence, uint32_t logical, uint32_t physical) {
  uint32_t block = physical == unbound ? sequence.bindings[logical] : physical;
  if (physical == unbound) {
    kv->map(uint32_t(sequence.slot) * blocks.size() + logical, unbound);
    --blocks[block].refs;
  } else {
    kv->map(uint32_t(sequence.slot) * blocks.size() + logical, physical);
    ++blocks[block].refs;
  }
  sequence.bindings[logical] = physical;
}
uint32_t Engine::acquireBlock() {
  uint32_t physical = physicalBlocks < blocks.size() ? physicalBlocks++ : unbound;
  if (physical == unbound) {
    auto victim = std::min_element(blocks.begin(), blocks.end(), [](const PhysicalBlock& a, const PhysicalBlock& b) {
      return (a.refs ? UINT64_MAX : a.touch) < (b.refs ? UINT64_MAX : b.touch);
    });
    if (victim->refs)
      throw std::runtime_error("KV block pool has no evictable capacity");
    physical = victim - blocks.begin();
    for (auto checkpoint = checkpointCache.begin(); checkpoint != checkpointCache.end();)
      if (std::find(checkpoint->second.bindings.begin(), checkpoint->second.bindings.end(), physical) != checkpoint->second.bindings.end())
        checkpoint = checkpointCache.erase(checkpoint);
      else
        ++checkpoint;
  }
  blocks[physical] = {};
  return physical;
}
void Engine::reserve(const Batch& batch) {
  for (uint32_t row = 0; row < batch.size; ++row) {
    Sequence& sequence = *batch.queries[row].sequence;
    for (uint32_t logical = sequence.kvValid / blockTokens; logical <= (sequence.kvValid + batch.queries[row].count - 1) / blockTokens; ++logical) {
      if (sequence.bindings[logical] == unbound) {
        kv->ensure(std::min<uint32_t>(physicalBlocks + 1, blocks.size()));
        bind(sequence, logical, acquireBlock());
      }
      blocks[sequence.bindings[logical]].touch = ++clock;
    }
  }
}
uint32_t Engine::lookupPrefix(Sequence& sequence, const int32_t* tokens, uint32_t length) {
  uint32_t limit = (length - 1) / blockTokens;
  uint64_t hash = 0;
  HybridCheckpoint* checkpoint = nullptr;
  for (uint32_t logical = 0; logical < limit; ++logical) {
    hash = hashBlock(hash, tokens + uint64_t(logical) * blockTokens);
    auto found = checkpointCache.find(hash);
    if (!((logical + 1) * blockTokens % gdnCheckpointTokens) && found != checkpointCache.end()) {
      checkpoint = &found->second;
      sequence.prefixHash = hash;
    }
  }
  if (!checkpoint)
    return 0;
  checkpoint->touch = ++clock;
  copyCheckpoint(*this, sequence, checkpoint->arena, true);
  for (uint32_t logical = 0; logical < checkpoint->bindings.size(); ++logical) {
    bind(sequence, logical, checkpoint->bindings[logical]);
    device.wait();
  }
  sequence.bank = 0;
  return checkpoint->bindings.size() * blockTokens;
}
void Engine::publishPrefix(Sequence& sequence, uint32_t oldValid, uint32_t valid) {
  try {
    uint32_t first = oldValid / blockTokens, count = valid / blockTokens;
    if (first >= count)
      return;
    for (uint32_t logical = first; logical < count; ++logical)
      sequence.prefixHash = hashBlock(sequence.prefixHash, sequence.request.data() + uint64_t(logical) * blockTokens);
    if (valid % gdnCheckpointTokens)
      return;
    if (checkpointCache.count(sequence.prefixHash)) {
      checkpointCache.at(sequence.prefixHash).touch = ++clock;
      return;
    }
    if (checkpointCache.size() == maxGdnCheckpoints) {
      auto victim = std::min_element(checkpointCache.begin(), checkpointCache.end(),
                                     [](const auto& a, const auto& b) { return a.second.touch < b.second.touch; });
      checkpointCache.erase(victim);
    }
    HybridCheckpoint checkpoint{device.empty(gdnCheckpointBytes + (drafter == Drafter::mtp ? 8192 : 0)),
                                {sequence.bindings.begin(), sequence.bindings.begin() + valid / blockTokens},
                                ++clock};
    copyCheckpoint(*this, sequence, checkpoint.arena, false);
    checkpointCache.emplace(sequence.prefixHash, std::move(checkpoint));
  } catch (...) {
  }
}
Sequence::Sequence(Engine& owner, const int32_t* stopTokens, uint32_t stopCount, float samplingTemperature, float samplingTopP,
                   int32_t samplingTopK, uint32_t drafts)
    : engine(owner), bindings(owner.blocks.size(), unbound), draftTokens(drafts), temperature(samplingTemperature), topP(samplingTopP),
      topK(samplingTopK) {
  std::lock_guard lock(engine.mutex);
  request.reserve(owner.maxContext + 1);
  stops.assign(stopTokens, stopTokens + stopCount);
  for (slot = 0; slot < maxBatchSequences && engine.sequences[slot]; ++slot) {
  }
  if (slot == maxBatchSequences)
    throw std::runtime_error("maximum live sequence count reached");
  engine.sequences[slot] = this;
}
Sequence::~Sequence() {
  std::unique_lock lock(engine.mutex);
  active = false;
  engine.condition.wait(lock, [&] { return !busy; });
  for (uint32_t logical = 0; logical < bindings.size() && bindings[logical] != unbound; ++logical)
    engine.bind(*this, logical, unbound);
  engine.sequences[slot] = nullptr;
}
} // namespace infeng::qwen35
