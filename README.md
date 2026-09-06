# Qwen3.5 Metal inference engine

MTLQwen is a specialized Qwen3.5-9B inference engine for Apple GPUs. Python tokenizes input and reads generated tokens. C++ owns the model, memory, sequence state, continuous scheduler, batching, prefix cache, speculative decoding, and Metal execution.

At most eight sequences are live and one model pass is in flight per engine.

## End-to-end flow

### 1. Create the engine

Python creates one native `Engine`. During construction, C++:

1. Maps the GGUF weights.
2. Compiles the Metal source libraries and builds the required pipelines.
3. Allocates model-wide control buffers, GDN state, RNG state, RoPE tables, and sparse virtual K/V buffers.
4. Starts the scheduler thread.

The state-bearing fields of the real `Engine` are:

```cpp
struct Engine {
  Device device;                              // Metal device, pipelines, commands, and allocations
  uint32_t maxContext;                        // Maximum computed tokens in one sequence
  std::unique_ptr<SparseKV> kv;               // Model-wide sparse K/V address space and physical heaps

  Tensor embedding, norm, rope, dflashRope;   // Target embeddings, final norm, and RoPE tables
  Linear head;                                // Shared language-model head
  std::array<Layer, targetLayers> layers;      // 32 target-model layers
  DrafterWeights draftModel;                  // MTP or DFlash weights when enabled
  Drafter drafter = Drafter::none;            // Active drafting strategy

  Tensor gdnStates[2], candidateStates;        // Persistent GDN banks and temporary speculative states
  Scratch workspace;                          // Grow-only reusable forward-pass scratch

  Tensor inputIds, batchKvValid;               // Packed input tokens and starting positions
  Tensor queryStartLoc, draftPositions;        // Packed-query boundaries and MTP positions
  Tensor sequenceSlots, stateBanks;            // Slot and current-bank metadata sent to kernels
  Tensor draftTokens, outputTokens;            // Draft proposals and target samples
  Tensor rng, mtpSeeds, logitRows;              // Per-slot RNG, banked MTP seeds, and sampled row indices

  std::vector<PhysicalBlock> blocks;            // Physical bundle reference and LRU metadata
  std::unordered_map<uint64_t, HybridCheckpoint> checkpointCache; // Prefix checkpoints by chained hash
  std::array<Sequence*, maxBatchSequences> sequences{};            // Eight live-sequence slots

  std::mutex mutex;                             // Protects sequence and scheduler state
  std::condition_variable condition;            // Wakes the scheduler and blocked readers
  std::thread worker;                            // Continuous scheduler thread
  bool closing = false, running = false;         // Engine shutdown and in-flight-pass state
  uint64_t clock = 0;                            // LRU clock for blocks and checkpoints
  uint32_t physicalBlocks = 0;                   // Physical bundle IDs introduced so far
  uint64_t parameterCount = 0, modelBytes = 0;   // Model information exposed to Python
};
```

### 2. Create a sequence and assign its slot

Python creates a sequence with fixed stop, sampling, and drafting configuration:

```python
sequence = engine.sequence(
    stop_token_ids=...,
    temperature=...,
    top_p=...,
    top_k=...,
    draft_tokens=...,
)
```

C++ immediately finds a free entry in `Engine.sequences`. Its index becomes `Sequence.slot`.

The real `Sequence` is:

```cpp
struct Sequence {
  Engine& engine;                    // Engine that owns the slot and executes this sequence
  std::vector<int32_t> request;      // Prompt and generated tokens in exact model order
  std::vector<uint32_t> bindings;    // Logical K/V block -> physical bundle ID
  std::vector<int32_t> stops;        // Token IDs that end generation
  std::exception_ptr error;          // Failure observed by Python on its next read

  uint64_t drafted = 0;              // Total proposals generated
  uint64_t accepted = 0;             // Total proposals accepted by the target
  uint64_t prefixHash = 0;            // Chained hash through the committed full blocks
  uint32_t kvValid = 0;               // Number of tokens committed to target K/V and GDN state

  uint32_t slot;                      // Index in Engine.sequences and all per-slot GPU state
  uint32_t bank = 0;                  // GDN bank containing the current committed state
  uint32_t draftTokens;               // Fixed requested proposal width
  float temperature, topP;            // Fixed sampling configuration
  int32_t topK;                        // Fixed sampling candidate limit
  bool active = false, busy = false;   // May continue; currently belongs to an in-flight batch
};
```

Creation allocates no physical K/V memory. `bindings` starts entirely `unbound`.

### 3. Append prompt tokens

Python tokenizes text when necessary and calls `Sequence.append()`. C++:

1. Appends the IDs to `Sequence.request`.
2. Sets `active = true`.
3. Wakes the scheduler.
4. Returns `request.size()` as Python's first output cursor.

The central invariant is:

```text
request[0 : kvValid]  has completed the target pass
request[kvValid : ]   still requires a target pass

0 <= kvValid <= request.size()
```

The sequence is ready for scheduling when:

```text
active && !busy && request.size() > kvValid
```

New prompt tokens may be appended only between passes:

```text
!active && !busy
```

### 4. Select the next batch

The scheduler waits until at least one sequence is ready, allows a short coalescing window, and scans all eight entries in `Engine.sequences`.

Every ready sequence enters the next `Batch` and becomes `busy`. Batch membership is then fixed; sequences activated during the pass remain available for the next scan.

The real `Batch` is:

```cpp
struct Batch {
  std::array<Query, maxBatchSequences> queries{}; // One query per selected sequence
  uint32_t size = 0;                              // Number of selected sequences
};
```

At this point each query contains only its `Sequence*`. The remaining fields are derived after prefix lookup.

### 5. Restore a cached prefix

Only a selected sequence with `kvValid == 0` performs prefix lookup.

The engine hashes consecutive 128-token blocks. A cache hit is usable only at a 512-token boundary and must leave at least one request token for a target pass.

For the deepest usable hit, C++:

1. Copies the cached GDN state into bank 0.
2. Restores the bank-0 MTP seed when MTP is active.
3. Maps the cached physical K/V bundles into this sequence's slot.
4. Records those bundle IDs in `Sequence.bindings`.
5. Sets `bank = 0` and advances `kvValid` to the checkpoint boundary.

Without a hit, `kvValid` remains zero.

### 6. Apply the scheduling policy and build each query

After prefix restoration, the scheduler computes:

```text
pending = request.size() - kvValid

pending > 1  prompt processing
pending == 1 ordinary decode or speculative verification
```

Every selected sequence first receives one target row. The remaining capacity is assigned in this order:

1. Speculative queries reserve their shared proposal width.
2. The rows still available are divided among prompt queries.
3. Ordinary decode queries remain one row wide.

For a two-sequence batch, this can produce:

```text
ordinary decode + long prompt       1 + 127 rows
2-proposal verification + long prompt 3 + 125 rows
```

The limits are:

```text
selected sequences <= 8
total target rows  <= 128
query end          <= next 512-token checkpoint boundary
```

The shared proposal width is the minimum allowed by the active drafter, each participating sequence's `draftTokens`, and its remaining room before the next checkpoint.

The real `Query` is:

```cpp
struct Query {
  Sequence* sequence = nullptr; // Persistent state advanced by this query
  uint32_t count = 1;           // Target rows: prompt chunk, one anchor, or anchor + proposals
  uint32_t logit = unbound;     // Offset in sampled results; unbound for an intermediate prompt chunk
  uint32_t state = unbound;     // First row in candidateStates; unbound when not speculative
};
```

`Query` and `Batch` exist only for this pass. Tokens, K/V bindings, GDN state, and sampling configuration remain owned by `Sequence` and `Engine`.

### 7. Reserve and bind sparse K/V memory

Now that every `Query.count` is known, `Engine::reserve()` calculates the 128-token logical blocks touched by the batch.

For each unbound logical block, C++:

1. Adds physical heap capacity if required.
2. Acquires a free or evictable physical K/V bundle.
3. Installs the Metal sparse mappings for every K/V layer.
4. Stores the bundle ID in `Sequence.bindings[logicalBlock]`.

```text
logicalBlock = tokenPosition / 128
virtualBlock = slot * blocksPerSlot + logicalBlock

bindings[logicalBlock] = physicalBundleId
```

`bindings` does not contain addresses. It is CPU metadata describing which physical bundle backs each logical block. `SparseKV::map()` installs the actual virtual-tile-to-physical-tile mappings used by Metal's MMU.

All required mappings exist before any model kernel runs.

### 8. Pack inputs and kernel metadata

C++ packs all query tokens into `Engine.inputIds` and builds one prefix sum:

```text
queryStartLoc[0] = 0
queryStartLoc[i + 1] = queryStartLoc[i] + query[i].count
```

For example:

```text
queryStartLoc = [0, 3, 4]

query 0 uses inputIds[0 : 3]
query 1 uses inputIds[3 : 4]
```

The complete per-pass metadata is:

```text
inputIds       packed input and proposal tokens
queryStartLoc  start and end of every packed query
batchKvValid   starting token position of every query
sequenceSlots  slot containing each query's sparse K/V, GDN state, and RNG
stateBanks     current Sequence.bank for every query
logitRows      packed rows that require sampling
draftPositions absolute positions used by MTP proposals
```

Kernels never receive a `Sequence*` or `Query*`. C++ translates their state into these flat buffers, tensor views, and scalar arguments.

### 9. Optionally produce draft proposals

Drafting is used only when a sequence has one pending anchor and a nonzero `draftTokens`. It does not advance `kvValid`.

#### Target only

The anchor goes directly to the target pass.

#### MTP

MTP starts from the anchor and the committed MTP seed. Its single layer generates proposals sequentially and stores them in `Engine.draftTokens`.

#### DFlash

DFlash packs each speculative query as an anchor followed by mask tokens. Its six layers process that packed input once and write the greedy proposals into `Engine.draftTokens`.

All three branches now join the same target pass.

### 10. Run the target model

The target input for each query is:

```text
prompt query      next Query.count request tokens
ordinary decode  one pending anchor
speculative       anchor followed by draft proposals
```

C++ binds the packed metadata, model weights, reusable scratch, GDN buffers, and sparse K/V resources to the kernels. The target pass:

1. Embeds the packed rows.
2. Runs 24 GDN layers and 8 full-attention layers.
3. Writes target K/V through the sequence's sparse virtual addresses.
4. Produces logits only for the rows marked by `logitRows`.
5. Samples with greedy argmax or the sequence's fixed sampling configuration.
6. Updates persistent drafter K/V and MTP seed data when applicable.

Attention uses `sequenceSlots` and token positions to address K/V. Metal's MMU follows the mappings installed in step 7.

GDN uses two persistent banks per slot:

```text
read bank  = Sequence.bank
write bank = 1 - Sequence.bank
```

### 11. Commit the result and select speculative candidates

Every successful query switches banks, whether it processed a prompt, ordinary decode, or speculative verification.

For non-speculative work, GDN kernels write the final state directly into:

```text
Engine.gdnStates[1 - Sequence.bank]
```

For speculative work, every verified position could become the accepted endpoint. GDN therefore writes one state after the anchor and one after each proposal into one engine-owned buffer shared by the speculative queries in that pass:

```text
Engine.candidateStates
```

Each candidate row contains the convolution and recurrent state of all 24 GDN layers. The buffer uses private GPU storage; “shared” here means that all queries use the same allocation.

The candidate location is carried by `Query.state`:

```text
candidate selected for A accepted proposals = Query.state + A
```

After target samples are compared with the proposals, C++ copies that candidate into `gdnStates[1 - bank]`. Rejected candidates are ignored.

`stateBanks` contains the current bank bit packed for GPU work. It does not contain candidate states or candidate offsets; those live in `candidateStates` and `Query.state` respectively.

C++ then commits each sequence:

```text
prompt or ordinary  kvValid += Query.count
speculative         kvValid += 1 anchor + accepted proposals

append sampled or accepted tokens to request
update RNG and speculative counters
bank ^= 1
```

After the flip, `bank` names the newly committed GDN state. MTP seeds follow the same bank bit.

The commit restores the invariant before readers are notified:

```text
request[0 : kvValid]  has valid target K/V and GDN state
request[kvValid : ]   is pending work
```

Complete 128-token blocks extend the sequence's prefix hash. At an exact 512-token boundary, the engine may publish a new prefix checkpoint.

### 12. Continue, stream, or stop

If the sequence remains active and `request.size() > kvValid`, the scheduler discovers it in the next scan. This single rule drives prompt continuation, ordinary decoding, and speculative verification; no explicit requeue exists.

Generated tokens are already appended to `Sequence.request`. Python calls:

```python
token = sequence.read(cursor)
cursor += 1
```

The cursor belongs to the caller. `read()` waits while the requested token has not appeared and generation is still active. It returns `None` after the sequence becomes inactive and the cursor has consumed every available token.

C++ clears `active` when:

```text
a stop token is generated
kvValid reaches maxContext
the caller cancels
execution fails
```

A frontend output limit is implemented by calling `cancel()` after reading the requested number of tokens.

### 13. Close the sequence

Closing a sequence:

1. Clears `active`.
2. Waits for its in-flight batch, if any.
3. Unmaps every bundle recorded in `Sequence.bindings`.
4. Decrements the physical bundle reference counts.
5. Clears `Engine.sequences[slot]`.

The slot can then be reused by another sequence.

## Appendix A: sparse K/V memory

Each slot receives enough sparse virtual address space for `maxContext`, but all slots share one physical pool:

```text
blocksPerSlot       = ceil(maxContext / 128)
virtualBlockCount   = 8 * blocksPerSlot
physicalBundleLimit = blocksPerSlot
```

The state-bearing fields of the real `SparseKV` are:

```cpp
class SparseKV {
  static constexpr uint64_t pageBytes = 256ull << 10; // One Metal sparse tile
  static constexpr uint64_t heapBytes = 64ull << 20;  // One placement heap
  Device& device;                                      // Installs sparse mappings
  uint32_t virtualBlocks;                              // Virtual blocks across all slots
  uint32_t maxPhysicalBlocks;                          // Shared physical bundle limit
  uint32_t layers;                                     // Target plus drafter K/V layers
  uint32_t tilesPerBlock;                              // Two tiles per layer: K and V
  uint32_t physicalBlocks = 0;                         // Bundles backed by current heaps
  uint32_t blocksPerHeap;                              // Complete bundles fitting in one heap
  uint64_t regionBytes;                                // Virtual bytes in one layer region
  Tensor resources[2];                                 // Sparse K resource and sparse V resource
  std::vector<NS::SharedPtr<MTL::Heap>> heaps;         // Physical 64 MiB placement heaps
};
```

One physical ID owns a complete 128-token bundle:

```text
target only  8 K + 8 V tiles                 = 4 MiB
MTP          target + 1 MTP K + 1 MTP V      = 4.5 MiB
DFlash       target + 6 DFlash K + 6 V       = 7 MiB
```

Physical bundle metadata is:

```cpp
struct PhysicalBlock {
  uint32_t refs = 0; // Number of live sequence bindings using this bundle
  uint64_t touch = 0; // Last-use clock for eviction
};
```

```text
refs > 0  bundle is pinned by a live sequence
refs == 0 bundle may be reused by the LRU allocator
```

If every physical bundle is pinned, allocation fails with `KV block pool has no evictable capacity`.

## Appendix B: prefix checkpoints

Attention K/V alone cannot restore Qwen3.5 because its GDN layers also carry recurrent state. A reusable prefix therefore needs both sparse bindings and a GDN checkpoint:

```cpp
struct HybridCheckpoint {
  Tensor arena;                   // Copied GDN state and optional MTP seed
  std::vector<uint32_t> bindings; // Physical bundle IDs through the boundary
  uint64_t touch = 0;             // Last-use clock for checkpoint eviction
};
```

Hashes are chained from block zero, so a cache entry identifies the entire preceding token prefix. The cache is local to one engine and retains up to eight LRU checkpoints.

## Appendix C: weights and pipelines

The engine maps each GGUF once and creates `Tensor` views into the mapped bytes. It does not copy individual weights.

During model construction:

1. Tensor names are resolved from the GGUF metadata.
2. Shapes and quantization types select the matching Metal linear kernels.
3. Pipeline states are created when first requested during model construction and cached by kernel name.
4. The 32 target layers are assembled from those tensor views and pipelines.
5. MTP layers come from the combined target/MTP GGUF; DFlash layers come from its separate GGUF.

The target topology is fixed:

```text
hidden size       4096
MLP size          12288
target layers     32
GDN layers        24
full-attention    8
```

## Appendix D: reusable and temporary buffers

`Engine.workspace` is one grow-only `Scratch` arena. It expands when a pass needs more packed rows, then its tensor views are reused by later passes.

```cpp
struct Scratch {
  bool decodeMode = false;                    // Selects decode or prefill linear kernels
  Tensor hidden[2], inputNorm, postNorm;       // Ping-pong hidden states and normalization outputs
  Tensor padInput, mlpGate, mlpUp, mlpMixed;   // Padded input and MLP intermediates
  Tensor attnQG, attnK, attnV, attnQRope;      // Attention projections and rotated queries
  Tensor attnKRope, attnOut, attnGated;        // Rotated keys and attention outputs
  Tensor attnPartials;                         // Partial attention reductions
  Tensor gdnMixed, gdnZ, gdnB, gdnG;           // GDN projections and scalar parameters
  Tensor gdnConvolved, gdnQ, gdnK, gdnV;       // GDN convolution and Q/K/V intermediates
  Tensor gdnDelta, mid;                        // Delta-rule output and layer residual
  Tensor targetHidden, dflashFeatures;         // Drafter inputs captured from the target
  Tensor targetLogits;                         // Reusable logits storage
  void ensure(Device&, uint32_t rows, Drafter); // Grows and partitions the arena when necessary
};
```

`Engine.candidateStates` is also grow-only, but contains meaningful data only during speculative verification. It is shared by every speculative query in the current pass; their `Query.state` offsets keep the ranges separate.

The control tensors are small shared-memory buffers written by C++ and read directly by Metal. Weights, GDN states, sparse K/V, candidates, and most scratch storage use private GPU memory.

## Appendix E: Python boundary

The complete Python-side generation loop is:

```python
from runtime import InferenceEngine

engine = InferenceEngine("weights/Qwen3.5-9B-UD-Q4_K_XL.gguf")
sequence = engine.sequence()

cursor = sequence.append([248045, 846, 198])
while (token := sequence.read(cursor)) is not None:
  cursor += 1
  print(token)

sequence.close()
engine.close()
```

Python owns only the native handle, optional tokenizer, stop-token copy used by the frontend, draft-width copy used for metrics, and its local read cursor. The authoritative token and inference state remains in the native `Sequence`.

## Source map

```text
runtime/inference.py          Python API and C ABI bindings
model/qwen.cpp                scheduler, query sizing, commit, and C API
model/qwen35.hpp              engine, sequence, batch, and model state
model/qwen35_weights.cpp      weight loading, pipelines, and engine construction
model/qwen_ops.cpp            draft and target forward passes
backend/metal/device.cpp      Metal commands, sparse allocation, and prefix cache
backend/metal/kernel/*.metal  GPU kernels
```

## Verification

```bash
pytest -q test/test_qwen35.py
python benchmarks/benchmark_qwen35.py --prompt-preset math --decode 64 --warmup 1 --iters 2
```

Keep batch size and proposal count modest on machines with 16 GB of unified memory.
