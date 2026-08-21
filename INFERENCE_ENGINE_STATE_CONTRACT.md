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
    request
    kv_valid
    state
}

state = pp | tg | eos
```

The fields mean:

- `sequence_id`: stable, unique control-plane identity for the sequence. Sequence IDs are not reused.
- `request`: frontend-provided request abstraction containing the mutable model-token sequence. That sequence contains the prompt, committed generated tokens, the current uncomputed anchor when in `tg`, and—temporarily—draft proposals. Other request metadata may remain encapsulated inside it.
- `kv_valid`: number of tokens at the beginning of `request` whose cache state is computed and committed for every enabled persistent cache group, including GDN state when present.
- `state`: routing state:
  - `pp`: prompt processing and prefix-cache lookup;
  - `tg`: token generation, optionally using a speculative drafter;
  - `eos`: terminal; the sequence must not be scheduled again.

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

The KV allocator owns:

```text
page_table[(sequence_id, logical_block_id)] = physical_block_id
ref_cnt[physical_block_id] = number_of_live_page_table_bindings
```

A physical block ID identifies a token-addressed cache bundle:

```text
CacheBundle {
    target_kv
    persistent_drafter_kv?  // present when the configured drafter needs token-addressed persistent state
}
```

The target and drafter tensors may use separate physical buffers, but the same physical block ID indexes both. They therefore share page-table bindings, reference counts, allocation, and eviction.

The page table describes placement and allocated capacity. It does not describe validity. `kv_valid` is the only validity boundary.

Consequently, the page table may contain mappings beyond `kv_valid`, including capacity retained after rejected speculative tokens. Those locations are invalid and may be overwritten by a later forward pass.

#### Hybrid GDN state

```text
GdnStateBuffer { recurrent, convolution }

gdn_state[sequence_id]: GdnStateBuffer
gdn_candidates[max_sequences_in_batch][K_max + 1]: GdnStateBuffer
```

Each live sequence owns one `gdn_state` containing the committed state after `request[:kv_valid]`. At `kv_valid == 0`, it contains the architecture-defined initial state.

`gdn_candidates` is engine-owned scratch. Batch construction assigns each selected sequence a row until that forward commits or fails. The row may represent a different sequence on the next forward.

Without speculation, the target writes the final query state to column zero. With speculation, column `i` contains the state after the anchor and `i` draft tokens:

```text
gdn_candidates[row][0] = final state after a non-speculative query
gdn_candidates[row][i] = state after anchor + first i draft tokens
speculative: 0 <= i <= draft_count <= K_max
```

After post-forward processing selects the commit boundary:

```text
column = new_kv_valid - old_kv_valid - 1 if is_draft else 0
copy(gdn_candidates[row][column], gdn_state[sequence_id[row]])
```

On failure, no copy occurs. The copy completes before `request`, `kv_valid`, and `state` are updated. The candidate row is reusable after commit or failure.

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

#### Hybrid GDN checkpoints

A hybrid model cannot resume from attention KV alone. Let:

```text
C = GDN checkpoint interval in tokens

C >= B
C % B == 0
```

`C` is fixed engine configuration. GDN checkpoints use the same chained `prefix_hash[i]` as the token-addressed cache:

```text
gdn_cache[prefix_hash[i]] = immutable {
    recurrent_state_after_block_i
    convolution_state_after_block_i
}
```

After commit, publish a checkpoint when the new boundary is aligned to `C`:

```text
if kv_valid > 0 and kv_valid % C == 0:
    i = kv_valid / B - 1
    gdn_cache[prefix_hash[i]] = copy(gdn_state[sequence_id])
```

`gdn_candidates` is never published.

A query must not cross a GDN checkpoint boundary:

```text
next_checkpoint = (floor(kv_valid / C) + 1) * C
kv_valid + token_count <= next_checkpoint
```

Prompt chunks and speculative proposals are limited by this boundary. Therefore `gdn_state[sequence_id]` is at the exact checkpoint position when a GDN checkpoint is published.

Walk attention hashes consecutively from block zero to the first miss or the prefix-hit limit. Select the deepest attention hit that also has a GDN checkpoint. For checkpoint block `i`:

```text
copy(gdn_cache[prefix_hash[i]], gdn_state[sequence_id])

for b in 0 ... i:
    physical = prefix_table[prefix_hash[b]]
    page_table[(sequence_id, b)] = physical
    ref_cnt[physical] += 1

kv_valid = (i + 1) * B
```

The copy, bindings, and `kv_valid` update form one logical operation. It completes before the sequence is scheduled.

Cached GDN checkpoints are immutable and are copied into the sequence's `gdn_state` on every hit. They are soft cache references and do not contribute to `ref_cnt`. Attention and GDN entries are allocated and evicted independently. A checkpoint cannot be evicted while its copy is in flight.

Attention hits after the selected GDN checkpoint are not bound and do not advance `kv_valid`.

If no matching GDN checkpoint exists, bind no attention blocks: `kv_valid` remains zero and the sequence keeps its initial `gdn_state`.

### 2.4 Reference-count contract

`ref_cnt` counts only live page-table bindings:

```text
ref_cnt[physical_block] =
    number of live (sequence_id, logical_block_id) entries
    whose value is physical_block
```

The following operations update it:

- binding a physical block into a sequence page table increments it;
- removing a sequence page-table binding decrements it;
- membership in `prefix_table` does not change it; binding a prefix hit into a sequence does;
- removing a prefix-table entry does not change it;
- `ref_cnt > 0` pins the physical block and makes it ineligible for replacement;
- `ref_cnt == 0` makes the block eligible for the allocator's replacement policy, but does not by itself evict or remove the cached prefix.

`ref_cnt` determines replacement eligibility, not replacement order. The allocator's eviction policy chooses among eligible physical blocks. Page-table and reference-count changes that transfer a live binding must be performed atomically from the allocator's point of view.

## 3. Meaning of `request + kv_valid`

`request` and `kv_valid` jointly encode the committed context and the current target-model query. No separate committed-token cursor or proposal list is required.

### 3.1 Prompt processing

During `pp`:

```text
request[:kv_valid] = cached or already-computed prompt tokens
request[kv_valid:] = prompt tokens still requiring target-model computation
```

Batch construction may process the remaining suffix in chunks.

### 3.2 Token generation without speculative decoding

Between generation steps, the normal `tg` invariant is:

```text
len(request) = kv_valid + 1
```

Therefore:

```text
request[:kv_valid] = committed cache context
request[kv_valid]  = anchor token without computed target KV
```

The anchor is the token whose forward pass produces the distribution for the next token.

### 3.3 Token generation with speculative decoding

The drafter receives the anchor and appends `K` proposed tokens temporarily:

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
state = eos
```

`terminal_end` is the exclusive token index immediately after that boundary. In this terminal case, do not append a replacement or bonus token.

Rejected proposal tokens are removed from `request`. Their physical KV storage may remain allocated, but it lies beyond `new_kv_valid` and is therefore invalid.

If the forward fails before commit, restore the steady form without additional rollback state:

```text
request = request[:kv_valid + 1]
```

For a hybrid model, leave `gdn_state[sequence_id]` unchanged. Its candidate row remains reusable.

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
    request
    kv_valid = 0
    state = pp
}
```

For a hybrid model, also allocate `gdn_state[sequence_id]` and initialize it to the architecture-defined initial state.

The request layer routes `pp` sequences to prefix lookup, `tg` sequences to the drafter or ordinary generation path, and never schedules `eos` sequences.

The frontend owns session identity and other frontend metadata. Each turn creates a new `sequence_id` and submits `(sequence_id, request)` to the engine. Previous turns are reused only through matching model-token prefixes; an `eos` sequence is never reactivated.

### 4.2 Prompt processing and prefix lookup

Logical block `b` covers:

```text
[b * B, (b + 1) * B)
```

Prefix lookup proceeds from logical block zero and stops at the first miss. Because hashes are chained, later blocks cannot be used after an earlier miss.

For a model whose persistent cache is entirely token-addressed, every hit advances validity by one block:

```text
physical = prefix_table[prefix_hash[b]]
page_table[(sequence_id, b)] = physical
ref_cnt[physical] += 1
kv_valid += B
```

These updates form one logical bind operation (logical <-> kv block).

For a hybrid model, attention hits alone do not establish complete target-model cache validity. Lookup instead follows Section 2.3 and advances `kv_valid` only to the deepest boundary having both consecutive attention hits and a matching GDN checkpoint.

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

The drafter uses `sequence_id` and the sequence page table to retrieve its persistent cache from the same physical cache bundle as the target KV. Any additional temporary input required by a particular drafter is defined in Section 5.

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

The same batch order is retained for `sequence_id`, `kv_valid`, and `is_draft`.

### 4.5 KV allocator

For each sequence in the execution batch, find the logical blocks touched by its scheduled query. For this we need `kv_valid[]` and `token_count[]`.

```text
query_positions = range(kv_valid, kv_valid + token_count)
required_logical_blocks = unique(position // B for position in query_positions)
```

Block reservation is all-or-nothing for the batch and completes before the target forward starts. If reservation fails, undo the new page-table bindings and reference-count changes, launch no forward, and leave every sequence's authoritative state unchanged.

For each required logical block:

```text
if page_table[(sequence_id, logical_block)] exists:
    reuse its physical block
else:
    physical_block = get_free_block_or_evict()
    page_table[(sequence_id, logical_block)] = physical_block
    ref_cnt[physical_block] += 1
```

Allocation, reuse, and eviction apply to the complete cache bundle. The target and persistent drafter cache groups cannot acquire different page-table mappings or lifetimes.

Hybrid GDN checkpoints are not token-addressed page-table blocks. They use the separate publication, restore, and eviction rules in Section 2.3.

KV from rejected draft tokens may be invalid, but its physical block remains mapped to the sequence. The allocator therefore reuses that mapping, and the next forward pass overwrites the invalid token positions.

The allocator changes physical capacity and page-table mappings. It does not change `request` or `kv_valid`.

When allocation requires replacing an existing physical block, the allocator:

```text
1. selects a victim for which ref_cnt[victim] == 0;
2. calls evict(victim);
3. evict(victim) removes the prefix-table entry pointing to victim,
   if one exists, and clears its hash metadata;
4. reuses the physical block for the new page-table binding;
5. increments ref_cnt[victim] for that new binding.
```

The prefix entry is not deleted merely because `ref_cnt` becomes zero. It is deleted only when its corresponding physical block is actually selected for eviction. Keeping prefix-table deletion inside the same eviction function guarantees that no hash entry points to a block after that block has been repurposed.

If a zero-reference cached block receives a prefix hit before eviction, the allocator binds it into the new sequence's page table and increments its `ref_cnt`; the block is no longer eligible for replacement.

On sequence release, remove every page-table binding, decrement its physical block's reference count, and release the sequence's `gdn_state` and temporary drafter state. Prefix-cache entries remain available under their normal eviction rules.

### 4.6 Target-model forward pass

The target model receives:

```text
(
    sequence_id[],
    batched_request,
    query_start_loc,
    kv_valid[],
    filtered_page_table,
    is_draft[],
    gdn_state?,
    gdn_candidates?,
)
```

`gdn_state` and `gdn_candidates` are passed only for a hybrid model. Batch row `r` reads `gdn_state[sequence_id[r]]` and writes `gdn_candidates[r]`.

`filtered_page_table` is the global page table restricted to the sequences active in this batch and ordered by batch row. Each row contains all page-table mappings for its sequence, including committed context blocks and blocks allocated beyond `kv_valid`.

For batch row `i`:
```text
query = batched_request[query_start_loc[i] : query_start_loc[i + 1]]
# if needed by RoPE
positions = range(kv_valid[i], kv_valid[i] + len(query))
```

Each query token attends to the `kv_valid[i]` cached tokens + its prefix inside the query (causal attention). Its position and the filtered page table determine where its KV is written
```python
# for batch row i
logical_block = position // B
block_offset = position % B
physical_block = filtered_page_table[i, logical_block]
```
The forward pass computes logits and writes KV for every query token. When speculation is enabled, it also exposes the target hidden states required to update the configured drafter as described in Section 5. It does not modify `request` or `kv_valid`; post-forward processing decides which written cache entries become committed. The committed interval is `[0, kv_valid)` for every token-addressed cache group.

### 4.7 Post-forward commit

Post-forward processing interprets the logits and selects the commit boundary without publishing it. Final `pp` uses the last query logit, ordinary `tg` uses the anchor logit, and speculative `tg` verifies the proposals. It then performs one logical per-sequence commit: finalize every enabled cache component through the selected boundary, update GDN state and drafter seed state when present, and publish `request`, `kv_valid`, and `state`. Batch construction, prefix publication, and sequence release cannot observe a partial commit.

#### Intermediate prompt chunk

```text
kv_valid += token_count
request unchanged
state remains pp
```

#### Final prompt chunk

```text
kv_valid = len(request)
request.append(first_generated_token)
state = tg, unless the token terminates the request
```

This establishes the steady generation invariant:

```text
len(request) = kv_valid + 1
```

#### Token generation without speculation

```text
kv_valid += 1
request.append(next_token)
```

Speculative generation applies the transition from Section 3.4.

#### Termination

When a stopping condition is committed:

```text
state = eos
```

An `eos` request may end with an uncomputed terminal token; this is valid because it will not be scheduled again. In all cases, tokens below `kv_valid` remain exactly the committed KV prefix.

After returning the terminal result, release the sequence as specified in Section 4.5.

## 5. Speculative drafter contract

### 5.1 Shared cache and validity

An enabled drafter stores persistent token-addressed state in the optional `persistent_drafter_kv` member of `CacheBundle`. It shares the target page-table address, `kv_valid` boundary, allocation, prefix binding, release, and eviction. There is no `drafter_kv_valid`.

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

The drafter retains this target hidden state as `mtp_seed[sequence_id]`. Drafting does not consume it: it survives until the corresponding target round commits, so a failed round can be retried. A successful commit replaces it with the selected target hidden state immediately before the new anchor; `eos` or sequence release discards it. `mtp_seed` is materialized drafter input, not a progress cursor.

When a separate MTP cache is used, MTP KV produced beyond `kv_valid` is tentative and is discarded after verification. Persistent MTP KV for newly committed positions is rebuilt from verified target outputs.

An MTP implementation that directly shares compatible target KV may omit the separate MTP cache group. The page-table and `kv_valid` contract remains unchanged.

## 6. Concurrency contract

Continuous batching uses one target-model forward at a time. Request intake and other CPU-side work may proceed concurrently, but the engine must enforce:

```text
at most one in-flight target-model batch per engine
```

This gives every in-flight sequence exclusive use of its `gdn_candidates` batch row until the forward commits or fails. Prefix-table, GDN-cache, page-table, and `ref_cnt` mutations are serialized through the allocator control path.

A scheduled item uses snapshots of `request`, `kv_valid`, its page-table bindings, and its `gdn_state` when present. Until that item commits or fails:

- an in-flight sequence cannot be released;
- shared physical prefix blocks remain protected by their reference counts;
- cached hybrid checkpoints remain immutable, and a restore source remains stable until its copy completes;
- the sequence's `gdn_state` remains unchanged while the target writes its candidate row;
- temporary KV writes do not advance the authoritative `kv_valid`.

The complete per-sequence commit must appear atomic to batch construction, prefix publication, and release:

```text
target and persistent drafter cache validity through new_kv_valid
gdn_state, when present
drafter seed state, when present
request
kv_valid
state
```

Allocator ownership changes must likewise be synchronized so that a block with `ref_cnt > 0` cannot be reused. When a zero-reference cached block is selected for reuse, its eviction function must remove the prefix-table entry before the block is repurposed.

## 7. Minimal-state constraints

To preserve the contract:

- use `kv_valid` as the only progress and validity cursor for target KV, persistent drafter KV, and GDN state;
- do not add a persistent proposal list—the temporary suffix of `request` already represents it;
- treat the page table as placement and capacity, never as validity;
- do not store positions, query lengths, or `query_start_loc` per sequence—they are batch derivations.
