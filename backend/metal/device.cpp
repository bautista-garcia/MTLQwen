#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "device.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace infeng::metal {
static std::string message(NS::Error* error) {
    auto* description = error ? error->localizedDescription() : nullptr;
    return description ? description->utf8String() : "Metal operation failed";
}
Buffer::~Buffer() { device->remove(metalBuffer); metalBuffer->release(); }
Device::Device(const std::filesystem::path& kernels, bool profile) {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    metalDevice = MTL::CreateSystemDefaultDevice();
    if (!metalDevice) throw std::runtime_error("Metal device unavailable");
    if (!metalDevice->supportsPlacementSparse()) throw std::runtime_error("placement sparse buffers unsupported");
    // create Metal 4 command submission, residency, and completion resources
    queue = metalDevice->newMTL4CommandQueue();
    allocator = metalDevice->newCommandAllocator();
    if (!queue || !allocator) throw std::runtime_error("MTL4 command resources unavailable");
    auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
    descriptor->setInitialCapacity(512);
    NS::Error* error = nullptr;
    residency = metalDevice->newResidencySet(descriptor.get(), &error);
    if (!residency) throw std::runtime_error(message(error));
    event = metalDevice->newSharedEvent();
    if (!event) throw std::runtime_error("shared event creation failed");
    queue->addResidencySet(residency);
    if (profile) {
        auto counters = NS::TransferPtr(MTL4::CounterHeapDescriptor::alloc()->init());
        counters->setType(MTL4::CounterHeapTypeTimestamp); counters->setCount(counterHeapEntries);
        counterHeap = metalDevice->newCounterHeap(counters.get(), &error);
        if (!counterHeap) throw std::runtime_error(message(error));
        timestampFrequency = metalDevice->queryTimestampFrequency();
    }
    // find and compile all Metal sources into libraries
    std::vector<std::filesystem::path> sources;
    for (const auto& entry : std::filesystem::directory_iterator(kernels))
        if (entry.path().extension() == ".metal") sources.push_back(entry.path());
    std::sort(sources.begin(), sources.end());
    for (const auto& path : sources) {
        std::ifstream file(path);
        std::stringstream stream;
        stream << file.rdbuf();
        auto source = NS::String::string(stream.str().c_str(), NS::UTF8StringEncoding);
        MTL::Library* library = metalDevice->newLibrary(source, nullptr, &error);
        if (!library) throw std::runtime_error(path.filename().string() + ": " + message(error));
        libraries.push_back(library);
    }
}
Device::~Device() {
    for (auto& [_, pipeline] : pipelines) pipeline->release();
    for (auto* library : libraries) library->release();
    if (counterHeap) counterHeap->release();
    event->release();
    queue->removeResidencySet(residency); residency->release();
    allocator->release(); queue->release(); metalDevice->release();
}
void Device::add(MTL::Allocation* allocation) { residency->addAllocation(allocation); residency->commit(); }
void Device::remove(MTL::Allocation* allocation) { residency->removeAllocation(allocation); residency->commit(); }
bool Device::wait() {
    uint64_t signal = ++eventValue; queue->signalEvent(event, signal);
    return event->waitUntilSignaledValue(signal, std::numeric_limits<uint64_t>::max());
}
void Device::recordKernel(const std::string& phase, const std::string& name, uint64_t gpuTimeNs) {
    auto found = std::find_if(kernelStats.begin(), kernelStats.end(), [&](const KernelCounter& counter) {
        return counter.phase == phase && counter.name == name;
    });
    if (found == kernelStats.end()) kernelStats.push_back({phase, name, gpuTimeNs, 1});
    else found->gpuTimeNs += gpuTimeNs, ++found->launches;
}
Tensor Device::empty(uint64_t bytes, bool shared) {
    MTL::Buffer* buffer = metalDevice->newBuffer(
        bytes, shared ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate);
    if (!buffer) throw std::runtime_error("buffer allocation failed");
    add(buffer);
    return {std::make_shared<Buffer>(this, buffer, bytes), 0, bytes};
}
Tensor Device::upload(const void* source, uint64_t bytes) {
    Tensor target = empty(bytes); write(target, source, bytes); return target;
}
void Device::write(const Tensor& destination, const void* source, uint64_t bytes) {
    if (bytes > destination.bytes) throw std::runtime_error("buffer write exceeds destination");
    if (destination.buffer->metalBuffer->storageMode() == MTL::StorageModeShared) {
        std::memcpy(static_cast<uint8_t*>(destination.buffer->metalBuffer->contents()) + destination.offset,
                    source, bytes);
        return;
    }
    // private buffers require a shared staging copy submitted before the host buffer is released
    MTL::Buffer* staging = metalDevice->newBuffer(source, bytes, MTL::ResourceStorageModeShared);
    if (!staging) throw std::runtime_error("staging allocation failed");
    add(staging);
    CommandBuffer commands(*this, 0);
    commands.copy(staging, 0, destination.buffer->metalBuffer, destination.offset, bytes);
    commands.commit();
    remove(staging); staging->release();
}
Pipeline* Device::pipeline(const std::string& name) {
    if (auto found = pipelines.find(name); found != pipelines.end()) return found->second;
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init()); MTL::Function* function = nullptr;
    auto string = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
    // find the kernel function across loaded libraries and cache its pipeline state
    for (auto* library : libraries) if ((function = library->newFunction(string))) break;
    if (!function) throw std::runtime_error("missing kernel " + name);
    NS::Error* error = nullptr;
    Pipeline* pipelineState = metalDevice->newComputePipelineState(function, &error);
    function->release();
    if (!pipelineState) throw std::runtime_error(name + ": " + message(error));
    pipelines.emplace(name, pipelineState);
    pipelineNames.emplace(pipelineState, name);
    return pipelineState;
}
CommandBuffer::CommandBuffer(Device& device, uint64_t constantBytes, bool profile, uint32_t dispatchCapacity,
                             const char* passPhase)
    : device(device), constants(constantBytes ? device.empty(constantBytes, true) : Tensor{}),
      phase(passPhase), profiling(profile && device.counterHeap) {
    metalCommandBuffer = device.metalDevice->newCommandBuffer();
    if (!metalCommandBuffer) throw std::runtime_error("MTL4 command buffer creation failed");
    // begin recording, attach resource residency, and open the compute encoder
    metalCommandBuffer->beginCommandBuffer(device.allocator);
    metalCommandBuffer->useResidencySet(device.residency);

    if (profiling) {
        counterHeap = device.counterHeap; counterLimit = 2 + 2 * dispatchCapacity;
        if (counterLimit > counterHeap->count())
            throw std::runtime_error("profiling forward exceeds the 4096-timestamp counter heap; use a shorter prefill");
        counterHeap->invalidateCounterRange(NS::Range::Make(0, counterLimit)); counterIndex = 2;
        metalCommandBuffer->writeTimestampIntoHeap(counterHeap, 0);
    }
    encoder = metalCommandBuffer->computeCommandEncoder();
    encoder->barrierAfterQueueStages(MTL::StageResourceState, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}
MTL4::ArgumentTable* CommandBuffer::table(uint32_t index) {
    if (index < tables.size()) return tables[index];
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(24); descriptor->setInitializeBindings(true);
    NS::Error* error = nullptr;
    auto* argumentTable = device.metalDevice->newArgumentTable(descriptor.get(), &error);
    if (!argumentTable) throw std::runtime_error(message(error));
    tables.push_back(argumentTable); return argumentTable;
}
void CommandBuffer::copy(MTL::Buffer* source, uint64_t sourceOffset, MTL::Buffer* destination,
                         uint64_t destinationOffset, uint64_t bytes) {
    encoder->copyFromBuffer(source, sourceOffset, destination, destinationOffset, bytes);
}
void CommandBuffer::submit() {
    // end, submit, wait for completion, resolve timestamps, and recycle command memory
    if (profiling) metalCommandBuffer->writeTimestampIntoHeap(counterHeap, 1);
    metalCommandBuffer->endCommandBuffer();
    const MTL4::CommandBuffer* commands[] = {metalCommandBuffer};
    device.queue->commit(commands, 1);
    if (!device.wait()) throw std::runtime_error("Metal event wait timed out");
    if (profiling) {
        NS::Data* data = counterHeap->resolveCounterRange(NS::Range::Make(0, counterIndex));
        auto* timestamps = static_cast<const MTL4::TimestampHeapEntry*>(data->bytes());
        device.stats.gpuTimeNs += device.timestampNanoseconds(timestamps[1].timestamp - timestamps[0].timestamp);
        ++device.stats.passes;
        for (const auto& profile : profiles)
            device.recordKernel(phase, profile.name, device.timestampNanoseconds(
                timestamps[profile.end].timestamp - timestamps[profile.start].timestamp));
    }
    device.allocator->reset();
}
CommandBuffer::~CommandBuffer() {
    for (auto* argumentTable : tables) argumentTable->release();
    metalCommandBuffer->release();
}
void CommandBuffer::commit() {
    encoder->endEncoding(); encoder = nullptr; submit();
}

Tensor SparseKV::makeBuffer(uint32_t regions) {
    uint64_t bytes = regionBytes * regions; MTL::Buffer* buffer = device.metalDevice->newBuffer(
        bytes, MTL::ResourceStorageModePrivate, MTL::SparsePageSize256);
    if (!buffer) throw std::runtime_error("sparse KV buffer allocation failed");
    device.add(buffer); return {std::make_shared<Buffer>(&device, buffer, bytes), 0, bytes};
}
SparseKV::SparseKV(Device& device, uint32_t virtualCount, uint32_t physicalCount, uint32_t layers, bool mtp)
    : device(device), virtualBlocks(virtualCount), maxPhysicalBlocks(physicalCount),
      tilesPerBlock(layers * 2 + (mtp ? 2 : 0)), blocksPerHeap(uint32_t(heapBytes / pageBytes) / tilesPerBlock),
      regionBytes(uint64_t(virtualBlocks) * pageBytes),
      resources{{makeBuffer(layers), layers, 0}, {makeBuffer(layers), layers, layers},
                {mtp ? makeBuffer(2) : Tensor{}, mtp ? 2u : 0u, layers * 2}} {}
void SparseKV::addHeap() {
    auto descriptor = NS::TransferPtr(MTL::HeapDescriptor::alloc()->init());
    descriptor->setType(MTL::HeapTypePlacement); descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setSize(heapBytes); descriptor->setSparsePageSize(MTL::SparsePageSize256);
    descriptor->setMaxCompatiblePlacementSparsePageSize(MTL::SparsePageSize256);
    MTL::Heap* heap = device.metalDevice->newHeap(descriptor.get());
    if (!heap) throw std::runtime_error("KV heap allocation failed");
    device.add(heap); heaps.push_back(heap); physicalBlocks += std::min(blocksPerHeap, maxPhysicalBlocks - physicalBlocks);
}
void SparseKV::ensure(uint32_t blocks) { while (physicalBlocks < blocks) addHeap(); }
void SparseKV::update(const Resource& resource, uint32_t virtualBlock, MTL::Heap* heap, uint32_t heapOffset) {
    auto mode = heap ? MTL::SparseTextureMappingModeMap : MTL::SparseTextureMappingModeUnmap;
    using Operation = MTL4::UpdateSparseBufferMappingOperation; auto operation = [&](uint32_t region) { return Operation{
        mode, NS::Range::Make(uint64_t(region) * virtualBlocks + virtualBlock, 1), heap ? heapOffset + region : 0}; };
    Operation operations[]{operation(0), operation(1), operation(2), operation(3), operation(4), operation(5),
                           operation(6), operation(7)};
    device.queue->updateBufferMappings(resource.buffer.buffer->metalBuffer, heap, operations, resource.regions);
}
void SparseKV::map(uint32_t virtualBlock, uint32_t physicalBlock) {
    MTL::Heap* heap = heaps[physicalBlock / blocksPerHeap]; uint32_t offset = physicalBlock % blocksPerHeap * tilesPerBlock;
    for (const Resource& resource : resources) if (resource.regions) update(resource, virtualBlock, heap,
                                                                             offset + resource.heapOffset);
}
void SparseKV::unmap(uint32_t virtualBlock) { for (const Resource& resource : resources) if (resource.regions)
    update(resource, virtualBlock, nullptr, 0); }
SparseKV::~SparseKV() {
    device.wait(); for (Resource& resource : resources) resource.buffer = {};
    for (auto* heap : heaps) { device.remove(heap); heap->release(); } }
}  // namespace infeng::metal
