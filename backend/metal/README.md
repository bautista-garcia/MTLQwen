# Qwen3.5 Metal inference engine

This document describes the checked-in engine. The state rules come from
[`INFERENCE_ENGINE_STATE_CONTRACT.md`](../../INFERENCE_ENGINE_STATE_CONTRACT.md); Section 10 records where the implementation deliberately or currently differs.

## 1. Scope and fixed limits

The engine is a specialized Qwen3.5-9B runtime for Apple GPUs with Metal 4 placement-sparse buffers.

```text
target layers               = 32
full-attention layers       = 8, at layers 3, 7, ..., 31
GDN layers                  = 24
hidden size                 = 4096
MLP size                    = 12288
vocabulary                  = 248320
KV block size               = 128 tokens
maximum context             = 65536 tokens
maximum live/batched slots  = 8
maximum packed target rows  = 128
maximum draft tokens        = 7
maximum GDN candidate rows  = 8 * (7 + 1) = 64
GDN checkpoint interval     = 512 tokens
maximum cached checkpoints  = 8
```

The active drafter is fixed when the model is created:

```text
none:    maximum 0 proposals
MTP:     maximum 4, default 2, tensors read from the target GGUF
DFlash:  maximum 7, default 7, separate draft GGUF required
```

DFlash ignores embedded MTP tensors. The implementation has no drafter auto-detection.

The execution path is [`runtime/inference.py`](../../runtime/inference.py) for request state and scheduling,
[`model/qwen.cpp`](../../model/qwen.cpp) for the native transaction, [`model/qwen_ops.cpp`](../../model/qwen_ops.cpp) for graph encoding, and
[`device.cpp`](device.cpp) plus [`kernel/`](kernel/) for Metal resources and compute.

## 2. State ownership

Python owns request policy:

```text
Session {
    sequence_id
    native handle
    request
    kv_valid
    phase                 // pp | tg
    outputs               // committed speculative tokens awaiting delivery
    sampling and stop configuration
    chat-formatting state
    closed
}
```

The native (c++) sequence owns only execution placement:

```text
Sequence {
    slot
    bindings[logical_block]
    bank
}
```

The model owns weights, pipelines, the shared physical-block pool, prefix hashes, hybrid checkpoints, both GDN banks, MTP seed banks, batch controls, RNG state, and reusable scratch.

The authoritative validity invariant is:

```text
request[:kv_valid] = tokens committed in target KV, enabled drafter KV, and GDN state
request[kv_valid:] = tokens not represented by committed persistent state

0 <= kv_valid <= len(request)
```

Bindings describe allocated address space. Writes beyond `kv_valid` are tentative even when the physical block remains mapped.

During ordinary generation:

```text
len(request) = kv_valid + 1
request[kv_valid] = the uncomputed anchor
```

If a stop token was accepted during verification, the turn may instead end with:

```text
len(request) = kv_valid
```

The next turn reuses the same session. If its first input token repeats that committed boundary token, `_attach` removes the overlap before appending the new prompt. Closing the session, rather than an `eos` phase, releases its native state.

## 3. Model and Metal initialization

Model creation performs the following work once:

1. Memory-map the target GGUF and, for DFlash, the draft GGUF. Weight tensors remain views into those mappings.
2. Compile every `.metal` source in `backend/metal/kernel` and lazily create pipelines by function name.
3. Select quantized linear pipelines from tensor type and shape. The implemented weight formats are Q4_K, Q5_K, Q6_K, Q8_0, and IQ4_XS.
4. Build the 32 target layers and the selected drafter.
5. Precompute target RoPE and, when needed, DFlash RoPE for the complete configured context.
6. Allocate shared CPU/GPU batch controls, two GDN banks per slot, two MTP seed banks when MTP is active, and per-slot RNG state.
7. Reserve sparse virtual KV address space. Placement heaps and physical pages are allocated only on demand.

Activations use one grow-only private arena. It is rounded to 32 rows and reused through non-overlapping tensor views. Linear kernels use the decode path when the largest query in the batch has at most five rows; larger queries use padded prefill kernels.

Every compute command buffer starts with a resource-state-to-dispatch barrier. Dispatches are device-visible to later dispatches. Submission is synchronous: the CPU waits on a shared event before command memory is reset or cache mappings can be removed.

## 4. Persistent cache

### 4.1 Sparse KV bundles

For configured context `T`:

```text
max_logical_blocks = ceil(T / 128)
virtual_blocks      = 8 * max_logical_blocks
physical_pool_limit = max_logical_blocks
virtual_block       = slot * max_logical_blocks + logical_block
```

The physical pool therefore holds one configured context worth of block bundles shared by all eight slots. It does not hold eight complete contexts.

One physical ID owns a complete 128-token bundle:

```text
target only:  8 K tiles + 8 V tiles                         = 4 MiB
MTP:          target tiles + 1 MTP K tile + 1 MTP V tile   = 4.5 MiB
DFlash:       target tiles + 6 DFlash K tiles + 6 V tiles  = 7 MiB
```

Each tile is one 256 KiB sparse page. Pages are backed by 64 MiB placement heaps.

The sparse resources are:

```text
target:  one 8-region K buffer and one 8-region V buffer
MTP:     one 2-region buffer containing K then V
DFlash:  one 6-region K buffer and one 6-region V buffer
```

At the 65,536-token maximum, one region spans 1 GiB across all slots. Splitting the resources keeps each sparse buffer within the device limit.

Attention binds the current layer's region view. The shader addresses:

```text
slot * (max_logical_blocks * 128) + token_position
```

Metal's MMU resolves the mapped page. No GPU page-table tensor or physical-block lookup is used by a shader.

### 4.2 Allocation and references

```text
refs[physical] = number of live native sequence bindings to physical
```

Prefix-table membership is not a reference. A zero-reference block remains a soft cache entry until the allocator needs it.

Before a batch executes, `reserve` finds every block touched by `[kv_valid, kv_valid + count)`. It first proves that enough unused or zero-reference blocks exist for the complete batch. Only then does it grow heaps, select LRU victims, install mappings, add bindings, and increment references.

The conditions are:

```text
refs > 0  => block cannot be evicted
refs == 0 => block is eligible, not immediately evicted
```

Eviction removes the prefix entry that still points to the victim, clears its hash metadata, and reuses the whole bundle. Target and drafter pages never have separate allocation lifetimes.

Releasing a sequence unmaps every bound virtual block and decrements its references. Completed GPU work is already synchronized by the engine lock and synchronous command submission.

### 4.3 GDN state

Each of the 24 GDN layers has two per-slot state banks:

```text
active   = bank
inactive = 1 - bank
```

An ordinary query reads the active convolution and recurrent state and writes its final state to the inactive bank. At `kv_valid == 0`, kernels synthesize zero initial state instead of reading old slot contents.

A speculative target query does not overwrite either bank. It writes one convolution and recurrent snapshot after each query token into grow-only candidate storage:

```text
state_start_loc[row + 1] = state_start_loc[row] + query_count[row]  if speculative
state_start_loc[row + 1] = state_start_loc[row]                     otherwise
```

Verification with `A` accepted proposals selects candidate column `A`, copies it to the inactive bank, and then flips `bank`. With at most eight rows and eight snapshots per speculative row, normal admission requires at most 64 candidate rows.

MTP seeds are double-buffered with the same bank selector. This makes the committed GDN state and the committed seed publish together.

## 5. Prefix cache

Only a new session performs prefix lookup. Blocks are hashed consecutively from block zero with an initial hash of zero. The model object is the cache namespace, so model weights and drafter configuration cannot share entries across engines.

Lookup leaves at least one prompt token for a target pass:

```text
complete_blocks_considered = floor((len(request) - 1) / 128)
```

Qwen3.5 is hybrid, so attention KV alone is insufficient. A hit is usable only at a 512-token boundary for which all preceding block hashes exist and a GDN checkpoint exists. The deepest such checkpoint is restored; later attention-only hits are ignored.

Restoration is synchronous and consists of:

```text
copy checkpoint GDN state into bank 0
copy checkpoint MTP seed into seed bank 0, when MTP is active
bind every sparse block through the checkpoint
set bank = 0
set kv_valid = checkpoint boundary
```

If no complete checkpoint exists, `kv_valid` remains zero and no prefix blocks are bound.

A prefix hit maps the same physical KV pages into the new slot and increments their references; it does not copy KV. Only the checkpoint's GDN state and optional MTP seed are copied.

After a successful commit, every newly completed 128-token block is published. At an exact 512-token boundary, the committed GDN state and optional MTP seed are copied into an immutable checkpoint. Checkpoints use a separate eight-entry LRU and do not hold physical KV references.

Prefix publication is an optimization: `publishPrefix` suppresses its own exceptions, so a failed publication does not fail an otherwise committed inference step.

## 6. Request scheduling and batch construction

`Session.forward` submits work to one scheduler thread and waits on a future. The scheduler selects at most eight requests that share the newest queued request's `(temperature, top_p, top_k)` tuple. Different sampling configurations cannot share one native batch because those values are batch-wide scalars.

After every pass, unfinished prompt requests return to the queue. Requests arriving during that pass can join the next batch, which is how chunked prefill and decode are continuously rebatched.

For every selected row:

```text
start_pos = kv_valid
```

The scheduler is intended to apply these conditions:

```text
ordinary decode count = 1
speculative count     = 1 + common_draft_count
prompt count          = a positive share of the remaining 128-row budget
query end             <= next 512-token checkpoint boundary
packed target rows    <= 128
```

The common draft count is the minimum requested count among eligible speculative rows, the drafter maximum, and the available checkpoint room. A row can request fewer proposals than another row, but all speculative rows selected for that pass execute the same width.

The native batch derives temporary metadata:

```text
query_start_loc[0] = 0
query_start_loc[i + 1] = query_start_loc[i] + query_count[i]

state_start_loc[0] = 0
state_start_loc[i + 1] = state_start_loc[i] + (query_count[i] if speculative else 0)
```

Tokens, `kv_valid`, slot IDs, bank IDs, and these prefix sums are packed into shared Metal buffers. Prompt, ordinary decode, and speculative verification rows use the same target forward path.

## 7. Forward transaction

One inference step is a prepare/execute/commit transaction.

### 7.1 Attach and prepare

For a fresh request:

```text
require len(input) > 0
request = input
kv_valid = deepest restorable prefix, or 0
phase = pp
```

A repeated prompt must preserve `request[:kv_valid]`. Active generation replaces the uncommitted suffix with the submitted anchor or next-turn tokens. If more than one token remains above `kv_valid`, the phase returns to `pp`.

The runtime derives query counts and flags, packs only the selected token suffix, and calls the native API while holding the engine lock. Native execution snapshots RNG state and reserves all required cache blocks before launching a drafter or target kernel.

### 7.2 Optional draft

Drafting starts only from a generation anchor and never advances `kv_valid`.

MTP runs one autoregressive step per proposal. The first step combines the committed target-hidden seed with the anchor embedding. Later steps combine the previous MTP hidden state with the previously proposed token. Each step uses greedy argmax. Its K/V writes at and above `kv_valid` are tentative.

DFlash packs each speculative row as:

```text
anchor + K mask positions, where 1 <= K <= 7
```

It runs six draft transformer layers once and greedily reads the mask-position logits. The first five layers use a 4,096-token sliding window; the last uses the full available context. Draft-layer K/V for the current proposal block is temporary.

### 7.3 Target model

The target pass:

1. Embeds all packed rows.
2. Runs 32 decoder layers. Every fourth layer uses full attention; the others use causal convolution plus the gated delta rule. Every layer ends with its residual MLP.
3. Writes target KV for every query token. Each attention row sees committed KV plus the causal prefix of its current query.
4. Produces logits only for requested rows: the final prompt/decode row, or every speculative verification row.
5. Materializes verified persistent drafter state from target outputs.

For MTP, the target pass writes MTP K/V for every query token. An ordinary row also writes its last target hidden state to the inactive seed bank. A speculative row keeps all target hidden states until verification selects the next seed.

For DFlash, target outputs after layers 1, 5, ..., 29 are concatenated, projected, and normalized. Each of the six draft layers projects that context into persistent DFlash K/V. Tentative draft state is therefore replaced by target-derived state at every verified position.

The forward writes cache and scratch but does not publish Python `request` or `kv_valid`.

### 7.4 Sampling and verification

```text
temperature <= 0 => greedy argmax
temperature > 0  => per-slot xorshift64 sampling
```

With `top_k == 0`, sampling scans the full vocabulary and does not apply `top_p`. With `1 <= top_k <= 64`, it constructs that top-k set and applies `top_p` inside it. The Python API rejects `top_p < 1` when `top_k` is omitted.

For speculation, the target samples one token after the anchor and after each proposal. Python accepts the longest prefix for which target samples equal drafter proposals.

For `A` accepted proposals:

```text
new_kv_valid = old_kv_valid + 1 + A
```

If no accepted proposal is a stop token:

```text
request = old committed prefix + anchor + accepted proposals + replacement_or_bonus
len(request) = new_kv_valid + 1
```

If an accepted proposal is a stop token:

```text
request = prefix through the first accepted stop token
len(request) = new_kv_valid
```

Rejected target, MTP, and DFlash entries remain mapped but invalid because they lie at or above the new boundary.

### 7.5 Commit and abort

Native commit performs, in order:

1. For speculative rows, copy candidate GDN column `A` and the selected MTP target hidden state into the inactive bank.
2. Rebase per-slot RNG state from its pre-forward snapshot using the accepted count and an intended next-token flag.
3. Flip the sequence bank.
4. Publish completed prefix blocks and an aligned checkpoint.

Only after native commit succeeds does Python assign the new `request`, `kv_valid`, and phase. The engine lock prevents scheduling or release from observing the native/Python handoff.

An abort restores RNG state and discards the pending native batch. It does not clear tentative KV, inactive GDN state, or candidate scratch; the unchanged bank and `kv_valid` make those writes invisible.

## 8. Speculative output delivery and turns

A speculative verification may commit several user-visible tokens in one target pass. Python returns the first and stores the rest in `Session.outputs`. Later `forward` calls validate the expected token order and drain this queue without running Metal. These are committed outputs, not unverified proposals.

`generate` retains the last returned token as the next anchor, applies the tokenizer's chat template only to the new turn, and preserves the native session across turns. A frontend stop ends generation but does not release cache state. `Session.close` is the only normal terminal transition.

## 9. Concurrency and failure conditions

There is at most one native batch in flight per model:

```text
scheduler worker + engine lock + synchronous Metal submission
```

The same lock serializes direct `forward_step`, prefix restore, native commit, and sequence release. Live sparse bindings pin physical blocks through `refs`. Candidate buffers grow before the target dispatch and are reused after commit or abort.

The main rejected conditions are:

```text
live sessions > 8
batch size outside 1..8
duplicate, closed, or foreign sessions in one batch
empty input
context outside 1..65536 or request beyond configured context
speculation without a drafter
proposal count outside the active drafter limit
DFlash without a separate draft GGUF
insufficient zero-reference physical blocks for the complete batch
stochastic top_k outside 0..64
top_p < 1 with top_k omitted
```

Reservation failure launches no forward. Validation or native failure leaves `kv_valid` and the active bank unchanged.

## 10. Differences from the state contract

The implementation preserves the contract's single `kv_valid` validity boundary. Its observable differences are:

- There is no stored `eos` phase. Python uses `closed`, and destruction immediately releases the native sequence.
- Draft proposals are held in native `draftTokens`; they are not temporarily appended to Python `request`. Only the verified result is published. `Session.outputs` is a separate queue for already committed tokens awaiting frontend delivery.
- Qwen3.5 always has non-token-addressed GDN state. Consequently, the implementation has no attention-only prefix-restore branch; every usable prefix ends at a 512-token hybrid checkpoint.
- MTP seed state has two banks selected by the same `Sequence.bank` as GDN, rather than one seed buffer updated by a separate copy protocol.
- Zero initial GDN and MTP state is established lazily by kernels when `kv_valid == 0`; both banks are not cleared when a slot is acquired.
- All speculative rows in a batch use one common proposal count. The contract permits a different count per row.
- The physical block pool contains `ceil(max_context / 128)` bundles total. Eight live sequences compete for that pool.
- Prefix publication is best-effort and suppresses errors; model-state commit remains authoritative.

Two current gaps require care:

- `_prepare` excludes a speculative row with only one token of checkpoint room when choosing the common draft width, but later marks every speculative generation row as drafted if another row enabled drafting. Such a mixed batch can cross that row's checkpoint boundary and can invalidate the intended 128-row budget calculation.
- RNG correction tests whether the block-sliced commit segment is longer than absolute `new_kv_valid`. That segment ends at `new_kv_valid`, so the test is always false. A non-terminal speculative commit advances the saved RNG by `A` draws instead of `A + 1`; terminal accepted-stop commits are unaffected because they consume only the `A` accepted draws. Greedy decoding is unaffected.

## 11. Benchmark and profiling

Target-only, MTP, and DFlash use the same benchmark:

```bash
python benchmarks/benchmark_qwen35.py --weights weights/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --prompt-preset math --decode 256 --iters 3

python benchmarks/benchmark_qwen35.py --weights weights/Qwen3.5-9B-UD-Q4_K_XL-MTP.gguf \
  --drafter mtp --speculative --prompt-preset math --decode 256 --iters 3

python benchmarks/benchmark_qwen35.py --weights weights/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --drafter dflash --draft-weights weights/qwen35-9b-dflash-Q4_K_M.gguf \
  --speculative --prompt-preset math --decode 256 --iters 3
```

Use chat-formatted tokens when measuring acceptance. Random vocabulary IDs measure kernel throughput, not a distribution on which either drafter was trained.

Kernel profiling uses precise Metal timestamps and reports them by `prefill`, `decode`, `verify`, `mtp_draft`, and `dflash_draft` phase:

```bash
python benchmarks/profile_qwen35.py --prefill 128 --decode 32
```

Timestamp markers add overhead. Use the wall-clock benchmark for final latency and throughput.
