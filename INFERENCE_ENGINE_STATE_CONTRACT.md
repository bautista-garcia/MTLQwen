# Inference Engine State Contract

## 1. Purpose and scope

This document defines the minimal state shared by an inference engine that supports:

- concurrent requests through continuous batching;
- chunked prompt processing and prefix caching;
- paged KV-cache allocation and sharing;
- speculative decoding during token generation;
- multi-turn requests submitted by a frontend.

The contract covers the state required to prepare, execute, and commit model-forward work. The concrete frontend representation and any request configuration remain encapsulated by `request`. When present, persistent drafter state shares the target KV allocator and does not introduce another progress cursor beside `kv_valid`.

At the engine boundary, `request` is the frontend-provided request abstraction. This contract requires it to expose the ordered token sequence consumed by the model and permit that sequence to be sliced, appended to, and truncated. The notation `request[...]` refers to that token sequence; it does not prescribe the request's concrete frontend representation.

## 2. Authoritative state

### 2.1 Per-sequence state

The authoritative per-sequence progress metadata is:

```text
SequenceState {
    sequence_id
    slot
    request
    kv_valid
    state
}

state = pp | tg | eos
```

The fields mean:

- `sequence_id`: stable, unique control-plane identity for one frontend conversation. It is retained across turns and is not reused.
- `slot`: bounded, reusable execution identity assigned while the sequence is live. Persistent GPU arrays and sparse virtual ranges are sized by the slot capacity, never by `sequence_id`.
- `request`: frontend-provided request abstraction containing the mutable model-token sequence. That sequence contains the prompt, committed generated tokens, the current uncomputed anchor when present in `tg`, and—temporarily—draft proposals. Other request metadata may remain encapsulated inside it.
- `kv_valid`: number of tokens at the beginning of `request` whose cache state is computed and committed for every enabled persistent cache group, including GDN state when present.
- `state`: routing state:
  - `pp`: prompt processing and prefix-cache lookup;
  - `tg`: token generation, optionally using a speculative drafter;
  - `eos`: the frontend has closed the conversation; the sequence must not be scheduled again.

The fundamental invariant is:

```text
request[:kv_valid] = tokens with valid, committed state in every enabled persistent cache group
request[kv_valid:] = tokens not yet represented by committed persistent cache state
```

Therefore:

```text
0 <= kv_valid <= len(request)
```

`kv_valid` is measured in tokens, not blocks. A physical block may be allocated while only part of it lies below `kv_valid`.

### 2.2 Global model-cache state

The engine fixes:

```text
B = token-addressed KV block size
K_max = maximum draft proposals per target query
max_sequences_in_batch = maximum execution batch rows

B >= 1
K_max >= 0
max_sequences_in_batch >= 1
```

The CPU allocator owns:

```text
logical_binding[(live_sequence, logical_block_id)] = physical_block_id
ref_cnt[physical_block_id] = number_of_live_logical_bindings
```

A physical block ID identifies a bundle of placement-heap tiles:

```text
CacheBundle {
    target_kv_tiles[plane]
    persistent_drafter_kv_tiles[plane]?  // present when the drafter needs token-addressed persistent state
}
```

For execution, each KV plane is one placement-sparse virtual buffer and each live logical binding has corresponding hardware mappings:

```text
virtual_block = slot * max_logical_blocks + logical_block_id
sparse_mapping[(plane, virtual_block)] = CacheBundle[physical_block_id].tile[plane]
```

The CPU logical binding is allocation and ownership metadata. The sparse mapping is the execution-time virtual-to-physical translation performed by Metal's MMU; it is not uploaded as a GPU page-table tensor. The target and drafter planes share the same physical block ID, logical binding, reference count, allocation, and eviction lifetime.

Bindings and sparse mappings describe placement and allocated capacity, not validity. `kv_valid` is the only validity boundary.

Consequently, a sequence may retain bindings and mappings beyond `kv_valid`, including capacity after rejected speculative tokens. Those locations are invalid and may be overwritten by a later forward pass.

#### Hybrid GDN state

```text
GdnStateBuffer { recurrent, convolution }

gdn_state[slot][2]: GdnStateBuffer
gdn_bank[slot]: 0 | 1
gdn_candidate_capacity: number of allocated GdnStateBuffer rows
```

Each live sequence owns two ordinary-private GDN buffers. The active bank contains the committed state after `request[:kv_valid]`; the other bank is inactive commit scratch:

```text
committed_gdn(slot) = gdn_state[slot][gdn_bank[slot]]
inactive_gdn(slot)  = gdn_state[slot][1 - gdn_bank[slot]]
```

At `kv_valid == 0`, the active bank contains the architecture-defined initial state. `gdn_bank` is CPU-owned publication metadata, not a second progress cursor.

An ordinary target forward reads the active bank and writes its final query state directly to the inactive bank. A successful commit flips `gdn_bank`; a failed forward does not. This publishes the new state without a GPU-to-GPU copy.

Speculative verification instead writes engine-owned candidate scratch. Candidate slices follow the packed
variable-length batch layout and are derived for the current forward:

```text
candidate_width[row] = draft_count[row] + 1 if is_draft[row] else 0
candidate_start_loc[0] = 0
candidate_start_loc[row + 1] = candidate_start_loc[row] + candidate_width[row]
required_candidate_rows = candidate_start_loc[batch_size]

gdn_candidate_capacity >= required_candidate_rows
gdn_candidate_capacity <= max_sequences_in_batch * (K_max + 1)
```

The arena grows on demand to the largest verification batch observed and need not shrink. It is not permanently allocated at the maximum size. Logical candidate column `i` contains the state after the anchor and `i` accepted draft tokens:

```text
gdn_candidates[row][i] = state after anchor + first i draft tokens
0 <= i <= draft_count <= K_max

physical_candidate_row = candidate_start_loc[row] + i
```

After verification selects the boundary:

```text
column = new_kv_valid - old_kv_valid - 1
copy(gdn_candidates[row][column], inactive_gdn(slot[row]))
flip gdn_bank[slot[row]]
```

The selected copy completes before the bank flip and the publication of `request`, `kv_valid`, and `state`. On failure, no copy or flip occurs. Candidate storage is reusable after commit or failure.

### 2.3 Global prefix-cache state

The token-addressed prefix cache owns:

```text
prefix_table[prefix_hash] = physical_block_id
```

For block `i`, its chained content hash is conceptually:

```text
prefix_hash[i] = hash(
    cache_namespace,
    prefix_hash[i - 1],
    request_tokens_in_block_i,
)
```

`cache_namespace` must distinguish any configuration that changes the resulting cache state, including the target model, active adapter, and configured drafter.

After any post-forward commit, publish every newly completed block containing valid entries for every enabled token-addressed cache group. Tentative speculative KV and partial blocks are never published.

A prefix-table entry is a soft cache reference: it makes a physical block discoverable by hash but does not pin the block and does not contribute to `ref_cnt`. A cached block with `ref_cnt == 0` remains in `prefix_table` until that physical block is selected for eviction.

#### Hybrid state checkpoints

A hybrid model cannot resume from attention KV alone. Let:

```text
C = hybrid state checkpoint interval in tokens

C >= B
C % B == 0
```

`C` is fixed engine configuration. Hybrid checkpoints use the same chained `prefix_hash[i]` as the token-addressed cache. They contain every non-token-addressed state component needed to resume at that boundary:

```text
HybridCheckpoint {
    gdn: GdnStateBuffer
    mtp_seed?  // present when the configured MTP drafter requires it
}

checkpoint_cache[prefix_hash[i]] = immutable HybridCheckpoint
```

Token-addressed target and persistent drafter KV are not copied into `HybridCheckpoint`. They are restored by recreating CPU logical bindings and their sparse mappings to the shared physical cache bundles. `mtp_seed` is included because it is one non-token-addressed per-sequence target hidden vector, not MTP KV and therefore not recoverable from those bindings.

After commit, publish a checkpoint when the new boundary is aligned to `C`:

```text
if kv_valid > 0 and kv_valid % C == 0:
    i = kv_valid / B - 1
    checkpoint = HybridCheckpoint {
        gdn = copy(committed_gdn(slot))
        mtp_seed = copy(mtp_seed[slot]) if required
    }
    checkpoint_cache[prefix_hash[i]] = checkpoint
```

`gdn_candidates` is never published.

A query must not cross a hybrid state checkpoint boundary:

```text
next_checkpoint = (floor(kv_valid / C) + 1) * C
kv_valid + token_count <= next_checkpoint
```

Prompt chunks and speculative proposals are limited by this boundary. Therefore the active GDN bank and any configured drafter seed are at the exact checkpoint position when a hybrid checkpoint is published.

Walk attention hashes consecutively from block zero to the first miss or the prefix-hit limit. Select the deepest attention hit that also has a complete hybrid checkpoint for every enabled non-token-addressed state component. For checkpoint block `i`:

```text
checkpoint = checkpoint_cache[prefix_hash[i]]
copy(checkpoint.gdn, gdn_state[slot][restored_bank])
copy(checkpoint.mtp_seed, mtp_seed[slot]) if present

for b in 0 ... i:
    physical = prefix_table[prefix_hash[b]]
    logical_binding[(live_sequence, b)] = physical
    map_sparse_bundle(slot, b, physical)
    ref_cnt[physical] += 1

gdn_bank[slot] = restored_bank
kv_valid = (i + 1) * B
```

The state copies, bank selection, bindings, and `kv_valid` update form one logical operation. It completes before the sequence is scheduled.

Cached hybrid checkpoints are immutable and are copied into the sequence's private GDN bank and drafter seed storage on every hit. The bank selection, logical bindings, sparse mappings, and `kv_valid` are then published as one logical operation. Checkpoints are soft cache references and do not contribute to `ref_cnt`. Token-addressed and hybrid-checkpoint entries are allocated and evicted independently. A checkpoint cannot be evicted while any component copy is in flight.

Attention hits after the selected hybrid checkpoint are not bound and do not advance `kv_valid`.

If no matching hybrid checkpoint exists, bind no attention blocks: `kv_valid` remains zero and the sequence keeps its initial active GDN bank and initial drafter seed state.

### 2.4 Reference-count contract

`ref_cnt` counts only live CPU logical bindings:

```text
ref_cnt[physical_block] =
    number of live (sequence, logical_block_id) entries
    whose value is physical_block
```

The following operations update it:

- binding a physical block to a sequence logical block and mapping its sparse ranges increments it;
- unmapping and removing a sequence logical binding decrements it;
- membership in `prefix_table` does not change it; binding a prefix hit into a sequence does;
- removing a prefix-table entry does not change it;
- `ref_cnt > 0` pins the physical block and makes it ineligible for replacement;
- `ref_cnt == 0` makes the block eligible for the allocator's replacement policy, but does not by itself evict or remove the cached prefix.

`ref_cnt` determines replacement eligibility, not replacement order. The allocator's eviction policy chooses among eligible physical blocks. Logical-binding, sparse-mapping, and reference-count changes that transfer a live binding must be performed atomically from the allocator's point of view.

## 3. Meaning of `request + kv_valid`

`request` and `kv_valid` jointly encode the committed context and the current target-model query. No separate committed-token cursor or proposal list is required.

### 3.1 Prompt processing

During `pp`:

```text
request[:kv_valid] = cached or already-computed prompt tokens
request[kv_valid:] = prompt tokens still requiring target-model computation
```

Batch construction may process the remaining suffix in chunks.

### 3.2 Token generation and turn boundaries

While actively decoding, the normal `tg` invariant is:

```text
len(request) = kv_valid + 1
```

Therefore:

```text
request[:kv_valid] = committed cache context
request[kv_valid]  = anchor token without computed target KV
```

The anchor is the token whose forward pass produces the distribution for the next token.

A frontend stopping condition ends the current turn, not the engine sequence. The sequence remains in `tg` with its KV bindings and mappings, GDN state, and drafter state retained. It may have either form:

```text
# The returned boundary is the next uncomputed anchor.
len(request) = kv_valid + 1

# Verification already committed the returned boundary.
len(request) = kv_valid
```

The frontend retains the returned boundary token when submitting the next turn. If it is already the final committed token, the request layer recognizes and elides that one-token overlap before appending the new model tokens. Otherwise it remains the first token of the appended prompt suffix. A turn-boundary sequence is dormant and is not scheduled until that input arrives. The appended suffix is processed through `pp`, after which the sequence returns to the active `tg` invariant.

### 3.3 Token generation with speculative decoding

Speculative decoding starts only from the active `len(request) = kv_valid + 1` form. The drafter receives the anchor and appends `K` proposed tokens temporarily:

```text
request = committed_cache_prefix + anchor + draft_1 + ... + draft_K
```

While proposals are attached:

```text
anchor_index        = kv_valid
draft_start         = kv_valid + 1
draft_count         = len(request) - kv_valid - 1
is_draft            = draft_count > 0
target_query        = request[kv_valid:]
target_query_length = len(request) - kv_valid
```

The target model processes the anchor and every proposal in one causal forward pass. `kv_valid` passed to that forward remains the pre-forward value: merely writing a KV entry does not make it committed.

### 3.4 Post-verification form

Let `A` be the number of accepted draft tokens, where `0 <= A <= K`. After verification:

```text
new_kv_valid = old_kv_valid + 1 + A
```

The retained target KV covers:

```text
anchor + accepted draft prefix
```

The request becomes:

```text
new_request =
    old_request[:new_kv_valid]
    + [next_token]
```

`next_token` is:

- the verifier-selected replacement when a proposal is rejected; or
- the bonus token sampled from the final target logit when all proposals are accepted.

Thus the engine returns to:

```text
len(new_request) = new_kv_valid + 1
```

If a stopping condition occurs inside the accepted draft prefix, commit only through the first terminal boundary:

```text
new_kv_valid = terminal_end
new_request = old_request[:terminal_end]
state = tg
```

`terminal_end` is the exclusive token index immediately after that boundary. This is the committed turn-boundary form from Section 3.2. Do not append a replacement or bonus token.

Rejected proposal tokens are removed from `request`. Their physical KV storage may remain allocated, but it lies beyond `new_kv_valid` and is therefore invalid.

If the forward fails before commit, restore the steady form without additional rollback state:

```text
request = request[:kv_valid + 1]
```

For a hybrid model, leave `gdn_bank[slot]` and its active committed state unchanged. Its candidate storage remains reusable.

## 4. Execution flow

### 4.1 Request layer

Input from the frontend:

```text
(sequence_id, request)
```

For a new sequence:
```text
require len(request) > 0

SequenceState {
    sequence_id
    slot = acquire_reusable_slot()
    request
    kv_valid = 0
    state = pp
}
```

For a hybrid model, also initialize both `gdn_state[slot]` banks and `gdn_bank[slot]` to the architecture-defined initial selection.

The request layer routes `pp` sequences to prefix lookup and active `tg` sequences to the drafter or ordinary generation path. A turn-boundary `tg` sequence with no uncomputed anchor remains dormant until the frontend appends input. The engine never schedules `eos` sequences.

The frontend owns conversation identity and other frontend metadata. Its first turn creates one engine `sequence_id`. Later turns append their ordered model tokens to the same sequence as described in Section 3.2; they do not replay the complete conversation into a new sequence. A live conversation therefore keeps its logical bindings, sparse mappings, slot, and persistent model state across turns. Releasing it returns the slot for reuse without reusing `sequence_id`.

Prefix caching remains available when a different sequence begins with matching model tokens. A sequence enters `eos` only when the frontend closes the conversation or declares a non-resumable terminal result. An `eos` sequence is never reactivated.

### 4.2 Prompt processing and prefix lookup

Logical block `b` covers:

```text
[b * B, (b + 1) * B)
```

Prefix lookup proceeds from logical block zero and stops at the first miss. Because hashes are chained, later blocks cannot be used after an earlier miss.

For a model whose persistent cache is entirely token-addressed, every hit advances validity by one block:

```text
physical = prefix_table[prefix_hash[b]]
logical_binding[(live_sequence, b)] = physical
map_sparse_bundle(slot, b, physical)
ref_cnt[physical] += 1
kv_valid += B
```

These updates form one logical bind operation (logical <-> kv block).

For a hybrid model, attention hits alone do not establish complete persistent-state validity. Lookup instead follows Section 2.3 and advances `kv_valid` only to the deepest boundary having both consecutive token-addressed hits and a complete matching hybrid checkpoint.

#### Prefix-hit limit

Prefix lookup always leaves at least one prompt token for a target forward pass. That pass produces the logits and any drafter seed state needed to start generation.

Under this minimal contract, the maximum cache-hit boundary is:

```text
max_prefix_hit_tokens = floor((len(request) - 1) / B) * B
```

This guarantees:

```text
kv_valid < len(request)
```

while the sequence remains in `pp`.

### 4.3 Drafter

When speculative decoding is disabled, `tg` bypasses the drafter and goes directly to batch construction. No proposals are appended, so `draft_count = 0`.

The minimal visible drafter input is:

```text
(sequence_id, request, kv_valid)

anchor = request[kv_valid]
```

The drafter uses the sequence's slot and sparse KV planes to retrieve its persistent cache from the same physical cache bundle as the target KV. `sequence_id` remains a control-plane identity. Any additional temporary input required by a particular drafter is defined in Section 5.

The number of proposals allowed in the round is:

```text
proposal_limit = K_max

if hybrid:
    next_checkpoint = (floor(kv_valid / C) + 1) * C
    proposal_limit = min(K_max, next_checkpoint - kv_valid - 1)
```

The drafter returns at most `proposal_limit` token IDs and temporarily appends them to `request`. If the limit is zero, the round bypasses the drafter. Drafting does not change target-model `kv_valid`.

After the append, `draft_count` and `is_draft` have the meanings defined in Section 3.3 and satisfy:

```text
0 <= draft_count <= proposal_limit
```

### 4.4 Batch construction

Batch construction chooses which sequences execute together and concatenates their pending tokens.

For each selected sequence:

```text
start_pos = kv_valid
```

First derive the maximum query length:

```text
max_query_tokens = len(request) - kv_valid

if hybrid:
    next_checkpoint = (floor(kv_valid / C) + 1) * C
    max_query_tokens = min(max_query_tokens, next_checkpoint - kv_valid)
```

For `pp`, the scheduler chooses a prompt chunk within that limit. For `tg`, all pending tokens are used:

```text
pp:               1 <= token_count <= max_query_tokens
tg (no spec dec): token_count = 1
tg (spec dec):    token_count = len(request) - kv_valid = 1 + draft_count <= max_query_tokens
```

```text
query = request[start_pos : start_pos + token_count]
```

Ordinary generation contributes one anchor. Speculative generation contributes the anchor followed by all temporarily attached proposals.

The scheduling policy chooses which queries enter the batch while respecting token and KV-memory budgets. It may favor `tg`/decode to reduce inter-token latency, favor `pp`/prefill to improve throughput, or mix both. The policy does not change the state contract.

The selected queries are concatenated in batch order:

```text
batched_request = concat(query_0, query_1, ..., query_n)
query_start_loc[0] = 0
query_start_loc[i + 1] = query_start_loc[i] + len(query_i)
```

The same batch order is retained for `slot`, `kv_valid`, and `is_draft`.

### 4.5 KV allocator

For each sequence in the execution batch, find the logical blocks touched by its scheduled query. For this we need `kv_valid[]` and `token_count[]`.

```text
query_positions = range(kv_valid, kv_valid + token_count)
required_logical_blocks = unique(position // B for position in query_positions)
```

Block reservation is all-or-nothing for the batch and completes before the target forward starts. Preflight enough physical capacity and complete tile bundles for the entire batch before installing any mapping or CPU binding. After preflight succeeds, install the mappings, bindings, and reference counts under the serialized allocator lock. If preflight fails, launch no forward and leave every sequence's authoritative state unchanged.

For each required logical block:

```text
if logical_binding[(live_sequence, logical_block)] exists:
    reuse its physical block
else:
    physical_block = get_free_block_or_evict()
    map_sparse_bundle(slot, logical_block, physical_block)
    logical_binding[(live_sequence, logical_block)] = physical_block
    ref_cnt[physical_block] += 1
```

Allocation, reuse, and eviction apply to the complete cache bundle. The target and persistent drafter planes cannot acquire different logical bindings or lifetimes.

Hybrid checkpoints are not token-addressed sparse KV blocks. They use the separate publication, restore, and eviction rules in Section 2.3.

KV from rejected draft tokens may be invalid, but its physical block remains mapped to the sequence. The allocator therefore reuses that mapping, and the next forward pass overwrites the invalid token positions.

The allocator changes physical capacity, CPU logical bindings, and sparse mappings. It does not change `request` or `kv_valid`.

When allocation requires replacing an existing physical block, the allocator:

```text
1. selects a victim for which ref_cnt[victim] == 0;
2. calls evict(victim);
3. evict(victim) removes the prefix-table entry pointing to victim,
   if one exists, and clears its hash metadata;
4. reuses the physical block for the new logical binding and sparse mappings;
5. increments ref_cnt[victim] for that new binding.
```

The prefix entry is not deleted merely because `ref_cnt` becomes zero. It is deleted only when its corresponding physical block is actually selected for eviction. Keeping prefix-table deletion inside the same eviction function guarantees that no hash entry points to a block after that block has been repurposed.

If a zero-reference cached block receives a prefix hit before eviction, the allocator binds and maps it into the new sequence's slot and increments its `ref_cnt`; the block is no longer eligible for replacement.

On sequence release, unmap every sparse virtual range, remove every CPU logical binding, decrement its physical block's reference count, and release both sequence GDN banks and temporary drafter state. Unmapping happens only after prior GPU commands complete. The engine-owned candidate arena remains available for later batches. Prefix-cache entries remain available under their normal eviction rules.

### 4.6 Target-model forward pass

The target model receives:

```text
(
    slot[],
    batched_request,
    query_start_loc,
    kv_valid[],
    is_draft[],
    gdn_state?,
    gdn_bank?,
    gdn_candidates?,
    candidate_start_loc?,
)
```

The GDN inputs are passed only for a hybrid model. Every row reads its active committed bank. An ordinary row writes its final state directly to the inactive bank. A speculative row writes `draft_count + 1` logical candidate columns into its assigned slice of the demand-sized candidate arena. Candidate capacity is ensured before the forward starts.

No logical-to-physical table is uploaded. The sparse mappings were established during reservation, and each attention dispatch binds only the current layer's sparse K/V plane pair. A resource-state-to-dispatch barrier orders mapping updates before the compute pass.

For batch row `i`:
```text
query = batched_request[query_start_loc[i] : query_start_loc[i + 1]]
# if needed by RoPE
positions = range(kv_valid[i], kv_valid[i] + len(query))
```

Each query token attends to the `kv_valid[i]` cached tokens plus its prefix inside the query (causal attention). The shader calculates the direct sparse virtual position:

```text
virtual_token = slot[i] * (max_logical_blocks * B) + position
```

Metal's MMU resolves the containing virtual block to the mapped physical tile. The shader performs no physical-block lookup or cache-plane arithmetic.
The forward pass computes logits and writes KV for every query token. When speculation is enabled, it also exposes the target hidden states required to update the configured drafter as described in Section 5. It does not modify `request` or `kv_valid`; post-forward processing decides which written cache entries become committed. The committed interval is `[0, kv_valid)` for every token-addressed cache group.

### 4.7 Post-forward commit

Post-forward processing interprets the logits and selects the commit boundary without publishing it. Final `pp` uses the last query logit, ordinary `tg` uses the anchor logit, and speculative `tg` verifies the proposals. It then performs one logical per-sequence commit: finalize every enabled cache component through the selected boundary, flip the GDN bank after its inactive bank is complete, update drafter seed state when present, and publish `request`, `kv_valid`, and `state`. Batch construction, prefix publication, and sequence release cannot observe a partial commit.

#### Intermediate prompt chunk

```text
flip gdn_bank, when present
kv_valid += token_count
request unchanged
state remains pp
```

#### Final prompt chunk

```text
flip gdn_bank, when present
kv_valid = len(request)
request.append(first_generated_token)
state = tg
```

This establishes the steady generation invariant:

```text
len(request) = kv_valid + 1
```

#### Token generation without speculation

```text
flip gdn_bank, when present
kv_valid += 1
request.append(next_token)
```

Speculative generation applies the transition from Section 3.4.

#### Turn boundary

When a configured generation stopping condition is returned, retain the sequence in `tg` in one of the forms from Section 3.2. Do not release any persistent state. The next frontend turn appends to the same sequence.

#### Sequence termination

Only closing the conversation or declaring its result non-resumable transitions the sequence:

```text
state = eos
```

An `eos` request may end with an uncomputed token; this is valid because it will not be scheduled again. In all cases, tokens below `kv_valid` remain exactly the committed KV prefix.

After the frontend closes the conversation or accepts the non-resumable result, release the sequence as specified in Section 4.5.

## 5. Speculative drafter contract

### 5.1 Shared cache and validity

An enabled drafter stores persistent token-addressed state in the optional `persistent_drafter_kv_tiles` member of `CacheBundle`. It shares the target logical binding, sparse virtual address, `kv_valid` boundary, allocation, prefix binding, release, and eviction. There is no `drafter_kv_valid`.

Temporary state created while producing proposals is not part of the cache bundle and does not become valid merely because it was written. After target verification, the persistent drafter cache is extended only through the same committed token boundary as target KV.

### 5.2 DFlash and DSpark

DFlash selects target decoder-layer outputs and builds one context feature per target token:

```text
H_selected = concat(H_target[layer] for layer in selected_layers)
H_ctx = norm(H_selected @ W_context)
```

For each drafter layer `j`, it stores:

```text
K_ctx[j] = K_projection[j](H_ctx)
V_ctx[j] = V_projection[j](H_ctx)
```

`K_ctx` and `V_ctx` form the persistent drafter cache group. The selected target hidden states and `H_ctx` may be discarded after these entries are materialized.

During drafting, the anchor and working draft tokens produce temporary drafter state:

```text
K_attention[j] = concat(K_ctx[j][:kv_valid], K_draft[j])
V_attention[j] = concat(V_ctx[j][:kv_valid], V_draft[j])
```

`K_draft` and `V_draft` are discarded after the drafting call. They are not persistent context, even when some proposals are later accepted. The accepted target-forward hidden states are instead projected into new `K_ctx` and `V_ctx` entries.

DSpark uses the same persistent cache contract. Its Markov and confidence heads do not add persistent per-sequence state; any recurrent Markov state exists only while constructing the current draft block.

### 5.3 MTP

MTP maintains its own persistent per-layer KV, stored as the drafter cache group in the shared physical blocks. A typical MTP step combines:

```text
target final-layer hidden state immediately before the input token
+ embedding(input token)
```

During `pp`, known prompt tokens let the engine build these MTP inputs as target hidden states become available and fill persistent MTP KV. After a target generation forward, the final committed target hidden state combines with the newly selected anchor in the same way.

The first input token is the anchor. Later speculative steps recursively use the previous MTP output hidden state and the previously proposed token.

Starting a draft round therefore requires both:

```text
persistent MTP KV for request[:kv_valid]
target hidden state immediately before request[kv_valid]
```

The drafter retains this target hidden state as `mtp_seed[slot]`. Drafting does not consume it: it survives until the corresponding target round commits, so a failed round can be retried. A successful commit replaces it with the selected target hidden state immediately before the new anchor. A frontend turn boundary retains it; `eos` or sequence release discards it. `mtp_seed` is materialized drafter input, not a progress cursor.

When hybrid prefix caching is enabled, `mtp_seed` is copied into and restored from the `HybridCheckpoint` defined in Section 2.3. Persistent MTP KV remains token-addressed state in `CacheBundle` and is restored through the shared logical bindings and sparse mappings.

When a separate MTP cache is used, MTP KV produced beyond `kv_valid` is tentative and is discarded after verification. Persistent MTP KV for newly committed positions is rebuilt from verified target outputs.

An MTP implementation that directly shares compatible target KV may omit the separate MTP planes. The logical-binding, sparse-mapping, and `kv_valid` contract remains unchanged.

## 6. Concurrency contract

Continuous batching uses one target-model forward at a time. Request intake and other CPU-side work may proceed concurrently, but the engine must enforce:

```text
at most one in-flight target-model batch per engine
```

This gives every speculative in-flight sequence exclusive use of its candidate-arena slice until the forward commits or fails. Prefix-table, GDN-cache, logical-binding, sparse-mapping, and `ref_cnt` mutations are serialized through the allocator control path. Candidate growth occurs before dispatch, while no target forward is in flight; an allocated arena is not resized while a command references it.

A scheduled item uses snapshots of `request`, `kv_valid`, its slot and logical bindings, and its `gdn_bank` when present. Until that item commits or fails:

- an in-flight sequence cannot be released;
- shared physical prefix blocks remain protected by their reference counts;
- cached hybrid checkpoints remain immutable, and a restore source remains stable until its copy completes;
- the active GDN bank and its selector remain unchanged while an ordinary forward writes the inactive bank or speculative verification writes candidate scratch;
- inactive GDN banks and candidate scratch are not authoritative until the selected state is complete and the bank flip is committed;
- temporary KV writes do not advance the authoritative `kv_valid`.

The complete per-sequence commit must appear atomic to batch construction, prefix publication, and release:

```text
target and persistent drafter cache validity through new_kv_valid
gdn_bank and the newly active gdn_state, when present
drafter seed state, when present
request
kv_valid
state
```

Allocator ownership changes must likewise be synchronized so that a block with `ref_cnt > 0` cannot be reused. When a zero-reference cached block is selected for reuse, its eviction function must remove the prefix-table entry before the block is repurposed.

## 7. Minimal-state constraints

To preserve the contract:

- use `kv_valid` as the only progress and validity cursor for target KV, persistent drafter KV, and GDN state;
- treat `gdn_bank` only as publication metadata for an already selected `kv_valid` boundary, never as another progress cursor;
- do not add a persistent proposal list—the temporary suffix of `request` already represents it;
- treat CPU logical bindings and hardware sparse mappings as placement and capacity, never as validity;
- do not store positions, query lengths, `query_start_loc`, or `candidate_start_loc` per sequence—they are batch derivations.
