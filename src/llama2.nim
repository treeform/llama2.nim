import cligen, std/tables, std/times, math, std/algorithm, std/strutils,
    std/streams, std/os

type
  Config = object
    dim: cint           # transformer dimension
    hidden_dim: cint    # for ffn layers
    n_layers: cint      # number of layers
    n_heads: cint       # number of query heads
    n_kv_heads: cint    # number of key/value heads (can be < query heads because of multiquery)
    vocab_size: cint    # vocabulary size, usually 256 (byte-level)
    seq_len: cint       # max sequence length

  TransformerWeights = object
    # token embedding table
    token_embedding_table: ptr[float32]    # (vocab_size, dim)
    # weights for rmsnorms
    rms_att_weight: ptr[float32] # (layer, dim) rmsnorm weights
    rms_ffn_weight: ptr[float32] # (layer, dim)
    # weights for matmuls. note dim == n_heads * head_size
    wq: ptr[float32] # (layer, dim, n_heads * head_size)
    wk: ptr[float32] # (layer, dim, n_kv_heads * head_size)
    wv: ptr[float32] # (layer, dim, n_kv_heads * head_size)
    wo: ptr[float32] # (layer, n_heads * head_size, dim)
    # weights for ffn
    w1: ptr[float32] # (layer, hidden_dim, dim)
    w2: ptr[float32] # (layer, dim, hidden_dim)
    w3: ptr[float32] # (layer, hidden_dim, dim)
    # final rmsnorm
    rms_final_weight: ptr[float32] # (dim,)
    # (optional) classifier weights for the logits, on the last layer
    wcls: ptr[float32]

  RunState = object
    # current wave of activations
    x: ptr[float32]   # activation at current time stamp (dim,)
    xb: ptr[float32]  # same, but inside a residual branch (dim,)
    xb2: ptr[float32] # an additional buffer just for convenience (dim,)
    hb: ptr[float32]  # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: ptr[float32] # buffer for hidden dimension in the ffn (hidden_dim,)
    q: ptr[float32]   # query (dim,)
    k: ptr[float32]   # key (dim,)
    v: ptr[float32]   # value (dim,)
    att: ptr[float32] # buffer for scores/attention values (n_heads, seq_len)
    logits: ptr[float32] # output logits
    # kv cache
    key_cache: ptr[float32]   # (layer, seq_len, dim)
    value_cache: ptr[float32] # (layer, seq_len, dim)

  Transformer = object
    config: Config              # the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights # the weights of the model
    state: RunState             # buffers for the "wave" of activations in the forward pass
    # some more state needed to properly clean up the memory mapping (sigh)
    fd: int                     # file descriptor for memory mapping
    data: ptr[float32]          # memory mapped data pointer
    file_size: int              # size of the checkpoint file in bytes

type
  TokenIndex = object
    str: cstring  # Equivalent of char* in Nim
    id: cint

  Tokenizer = object
    vocab: ptr[ptr[char]] # Equivalent of char** in Nim
    vocab_scores: ptr[float32]
    sorted_vocab: ptr[TokenIndex]
    vocab_size: cint
    max_token_length: uint32      # Equivalent of unsigned int in Nim
    byte_pieces: array[512, uint8] # Equivalent of unsigned char[512] in Nim

type
  ProbIndex = object
    prob: float32  # float in C
    index: cint    # int in C

  Sampler = object
    vocab_size: cint
    probindex: ptr[ProbIndex] # equivalent of ProbIndex* in C
    temperature: float32
    topp: float32
    rng_state: uint64   # equivalent of unsigned long long in C


proc newTransformer(t: var Transformer, checkpointPath: string) =
  var f = newFileStream(checkpointPath)
  var c: Config
  f.read(c)
  t.config = c

  if t.config.vocab_size < 0:
    quit("Negative t.config.vocab_size?")
  # let
  #   shared_weights =
  #     if t.config.vocab_size > 0:
  #       true
  #     else:
  #       false
  # # let
  # #   t.config.vocab_size = abs(t.config.vocab_size)
  let
    shared_weights = true

  t.file_size = getFileSize(checkpointPath).int

  proc read_floats(count: int): ptr[float32] =
    let buffer = alloc0(count * 4)
    var bytes = f.readData(buffer, count * 4)
    if bytes != count * 4:
      quit("read error")
    return cast[ptr[float32]](buffer)

  t.weights.token_embedding_table = read_floats(c.vocab_size * c.dim)
  t.weights.rms_att_weight = read_floats(c.n_layers * c.dim)
  t.weights.wq = read_floats(c.n_layers * c.dim * c.dim)
  t.weights.wk = read_floats(c.n_layers * c.dim * c.dim)
  t.weights.wv = read_floats(c.n_layers * c.dim * c.dim)
  t.weights.wo = read_floats(c.n_layers * c.dim * c.dim)
  t.weights.rms_ffn_weight = read_floats(c.n_layers * c.dim)
  t.weights.w1 = read_floats(c.n_layers * c.dim * c.hidden_dim)
  t.weights.w2 = read_floats(c.n_layers * c.hidden_dim * c.dim)
  t.weights.w3 = read_floats(c.n_layers * c.dim * c.hidden_dim)
  t.weights.rms_final_weight = read_floats(c.dim)
  #t.weights.freq_cis_real
  discard read_floats(c.seq_len * (c.dim div c.n_heads) div 2)
  #t.weights.freq_cis_imag =
  discard read_floats(c.seq_len * (c.dim div c.n_heads) div 2)
  if shared_weights:
    t.weights.wcls = t.weights.token_embedding_table
  else:
    if (t.file_size - f.getPosition()) div 4 <= 0:
      quit("wcls size is invalid!")
    t.weights.wcls = read_floats((t.file_size - f.getPosition()) div 4)

  # Allocate run state
  let kv_dim = (c.dim * c.n_kv_heads) div c.n_heads
  t.state.x = cast[ptr float32](alloc0(c.dim * sizeof(float32)))
  t.state.xb = cast[ptr float32](alloc0(c.dim * sizeof(float32)))
  t.state.xb2 = cast[ptr float32](alloc0(c.dim * sizeof(float32)))
  t.state.hb = cast[ptr float32](alloc0(c.hidden_dim * sizeof(float32)))
  t.state.hb2 = cast[ptr float32](alloc0(c.hidden_dim * sizeof(float32)))
  t.state.q = cast[ptr float32](alloc0(c.dim * sizeof(float32)))
  t.state.k = cast[ptr float32](alloc0(kv_dim * sizeof(float32)))
  t.state.v = cast[ptr float32](alloc0(kv_dim * sizeof(float32)))
  t.state.att = cast[ptr float32](alloc0(c.n_heads * c.seq_len * sizeof(float32)))
  t.state.logits = cast[ptr float32](alloc0(c.vocab_size * sizeof(float32)))
  t.state.key_cache = cast[ptr float32](alloc0(c.n_layers * c.seq_len * kv_dim * sizeof(float32)))
  t.state.value_cache = cast[ptr float32](alloc0(c.n_layers * c.seq_len * kv_dim * sizeof(float32)))



# proc build_transformer(t: ptr[Transformer], checkpoint_path: cstring) {.importc.}
#proc build_tokenizer(t: ptr[Tokenizer], tokenizer_path: cstring, vocab_size: int) {.importc.}
# proc build_sampler(sampler: ptr[Sampler], vocab_size: cint, temperature: float32, topp: float32, rng_seed: uint64) {.importc.}
# proc generate(transformer: ptr Transformer, tokenizer: ptr Tokenizer, sampler: ptr Sampler, prompt: cstring, steps: cint) {.importc.}

#proc chat(transformer: ptr Transformer, tokenizer: ptr Tokenizer, sampler: ptr Sampler, cli_user_prompt: cstring, cli_system_prompt: cstring, steps: cint) {.importc.}
#proc encode(t: ptr Tokenizer, text: cstring, bos: int8, eos: int8, tokens: ptr cint, n_tokens: ptr cint) {.importc.}
#proc decode(t: ptr Tokenizer, prev_token: cint, token: cint): cstring {.importc.}

proc `+`[T](p: ptr[T], n: SomeInteger): ptr[T] =
  cast[ptr[T]](cast[uint64](p) + n.uint64 * sizeof(T).uint64)

proc `[]`[T](p: ptr[T], n: SomeInteger): T =
  let p2 = p + n
  return p2[]

proc `[]=`[T](p: ptr[T], n: SomeInteger, v: T) =
  let p2 = p + n
  p2[] = v


proc newTokenizer(t: var Tokenizer, tokenizer_path: string, vocab_size: cint) =
  let f = newFileStream(tokenizer_path)
  t.vocab_size = vocab_size

  t.vocab = cast[ptr ptr char](alloc0(vocab_size * sizeof(ptr char)))
  t.vocab_scores = cast[ptr float32](alloc0(vocab_size * sizeof(float32)))

  for i in 0..255:
    t.byte_pieces[i * 2] = i.uint8
    t.byte_pieces[i * 2 + 1] = 0.uint8

  t.max_token_length = f.readUInt32()
  #echo "max_token_length ", max_token_length
  for i in 0 ..< vocab_size:
    #echo i
    t.vocab_scores[i] = f.readFloat32()
    let len = f.readInt32()
    #echo "len ", len
    let bstr = f.readStr(len)
    #echo bstr
    t.vocab[i] = cast[ptr char](alloc0(len+1))
    for j in 0 ..< len:
      t.vocab[i][j] = bstr[j]

proc decode(t: ptr Tokenizer, prev_token: cint, token: cint): string =
  var piece = t.vocab[token]
  for i in 0 .. 32:
    if piece[i] == '\0':
      break
    result.add piece[i]

  if prev_token == 1 and result[0] == ' ':
    result = result[1 .. ^1]

  if result == "<0x0A>":
    result = "\n"

proc getToken(t: ptr Tokenizer, tokenNum: cint): string =
  var token = ""
  var piece = t.vocab[tokenNum]
  for i in 0 .. 32:
    if piece[i] == '\0':
      break
    token.add piece[i]
  return token

proc str_lookup(t: ptr Tokenizer, s: string): cint =
  for tokenId in 0 ..< t.vocab_size:
    var token = t.getToken(tokenId)
    if token == s:
      return tokenId.cint
  return -1

proc encode(t: ptr Tokenizer, text: string): seq[cint] =

  var tokens: seq[cint]
  tokens.add 1

  if text == "":
    return tokens

  var text = " " & text

  # First encode every individual character in the input text
  for c in text:
    let id = t.str_lookup($c)
    if id == -1:
      quit("Error encoding")
    tokens.add(id.cint)

  # Merge the best consecutive pair each iteration, according to the scores in vocab_scores
  while true:
    var
      best_score = float32.low
      best_id = -1.cint
      best_idx = -1.cint

    for i in 0 ..< tokens.len - 1:
      # Check if we can merge the pair (tokens[i], tokens[i+1])
      var str = t.getToken(tokens[i]) & t.getToken(tokens[i + 1])
      let id = t.str_lookup(str)
      #echo "id ", id
      if id != -1 and t.vocab_scores[id] > best_score:
        # This merge pair exists in vocab! Record its score and position
        best_score = t.vocab_scores[id]
        best_id = id
        best_idx = i.cint

    if best_idx == -1:
      break  # We couldn't find any more pairs to merge, so we're done

    # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id
    # Delete token at position best_idx+1, shift the entire sequence back 1
    tokens.delete(best_idx + 1)
    #echo "merge", tokens[best_idx]

  return tokens

#proc forward(transformer: ptr Transformer, token: cint, pos: cint): ptr float32 {.importc.}
#proc sample(sampler: ptr Sampler, logits: ptr float32): cint {.importc.}
#proc rmsnorm(o: ptr float32, x: ptr float32, weight: ptr float32, size: cint) {.importc.}
#proc softmax(x: ptr float32, size: cint) {.importc.}
#proc matmul(xout: ptr float32, x: ptr float32, w: ptr float32, n: cint, d: cint) {.importc.}

template powf(a, b: float32): float32 = pow(a, b)
template cosf(a: float32): float32 = cos(a)
template sinf(a: float32): float32 = sin(a)
template sqrtf(a: float32): float32 = sqrt(a)
template expf(a: float32): float32 = exp(a)

proc rmsnorm(o: ptr float32, x: ptr float32, weight: ptr float32, size: cint) =
  # Calculate sum of squares
  var ss = 0.0'f32
  for j in 0 ..< size:
    ss += x[j] * x[j]
  ss /= size.float32
  ss += 1e-5'f32
  ss = 1.0'f32 / sqrtf(ss)
  # Normalize and scale
  for j in 0 ..< size:
    o[j] = weight[j] * (ss * x[j])

proc softmax(x: ptr float32, size: cint) =
  # Find max value (for numerical stability)
  var maxVal = x[0]
  for i in 1 ..< size:
    if x[i] > maxVal:
      maxVal = x[i]

  # exp and sum
  var sum = 0.0'f32
  for i in 0 ..< size:
    x[i] = expf(x[i] - maxVal)
    sum += x[i]

  # Normalize
  for i in 0 ..< size:
    x[i] = x[i] / sum

proc matmul(xout: ptr float32, x: ptr float32, w: ptr float32, n: cint, d: cint) =
  # W (d,n) @ x (n,) -> xout (d,)
  # by far the most amount of time is spent inside this little function

  # Parallelize outer loop
  # {.parallel.}
  for i in 0 ..< d:
    var val = 0.0'f32
    for j in 0 ..< n:
      val += w[i * n + j] * x[j]
    xout[i] = val

proc forward(transformer: ptr Transformer, token: cint, pos: cint): ptr float32 =
  # Convenience variables
  let p = addr transformer.config
  let w = addr transformer.weights
  let s = addr transformer.state
  let x = s.x
  let dim = p.dim
  let kv_dim = (p.dim * p.n_kv_heads) div p.n_heads
  let kv_mul = p.n_heads div p.n_kv_heads  # Integer multiplier of the kv sharing in multiquery
  let hidden_dim = p.hidden_dim
  let head_size = dim div p.n_heads

  # copy the token embedding into x
  let contentRow = w.token_embedding_table + token * dim
  copyMem(x, contentRow, dim * sizeof(x[]))

  # Forward all the layers
  for l in 0 ..< p.n_layers:

    # Attention rmsnorm
    rmsnorm(s.xb, x, w.rms_att_weight + l*dim, dim)

    # QKV matmuls for this position
    matmul(s.q, s.xb, w.wq + l*dim*dim, dim, dim)
    matmul(s.k, s.xb, w.wk + l*dim*kv_dim, dim, kv_dim)
    matmul(s.v, s.xb, w.wv + l*dim*kv_dim, dim, kv_dim)

    # RoPE relative positional encoding: complex-valued rotate q and k in each head
    for i in countup(0, dim - 1, 2):
      let headDim = i mod headSize
      let freq = 1.0'f32 / powf(10000.0'f32, float32(headDim) / float32(headSize))
      let val = float32(pos) * freq
      let fcr = cosf(val)
      let fci = sinf(val)
      let rotn = if i < kvDim: 2 else: 1  # How many vectors? 2 = q & k, 1 = q only

      for v in 0..<rotn:
        let vec = if v == 0: s.q else: s.k  # The vector to rotate (query or key)
        let v0 = vec[i]
        let v1 = vec[i + 1]
        vec[i] = v0 * fcr - v1 * fci
        vec[i + 1] = v0 * fci + v1 * fcr

    # Save key, value at this time step (pos) to our kv cache
    let loff = l * p.seqLen * kvDim  # kv cache layer offset for convenience
    let keyCacheRow = s.keyCache + loff + pos * kvDim
    let valueCacheRow = s.valueCache + loff + pos * kvDim
    copyMem(keyCacheRow, s.k, kvDim * sizeof(float32))
    copyMem(valueCacheRow, s.v, kvDim * sizeof(float32))

    # Multihead attention. Iterate over all heads
    # var h: cint
    # pragma omp parallel for private(h)
    for h in 0..<p.nHeads:
      # Get the query vector for this head
      let q = s.q + h * headSize
      # Attention scores for this head
      let att = s.att + h * p.seqLen
      # Iterate over all timesteps, including the current one
      for t in 0..pos:
        # Get the key vector for this head and at this timestep
        let k = s.keyCache + loff + t * kvDim + (h div kvMul) * headSize
        # Calculate the attention score as the dot product of q and k
        var score: float32 = 0.0
        for i in 0..<headSize:
            score += q[i] * k[i]
        score /= sqrt(float32(headSize))
        # Save the score to the attention buffer
        att[t] = score

      # Softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 1)

      # Weighted sum of the values, store back into xb
      let xb = s.xb + h * headSize
      zeroMem(xb, headSize * sizeof(float32))
      for t in 0..pos:
        # Get the value vector for this head and at this timestep
        let v = s.valueCache + loff + t * kvDim + (h div kvMul) * headSize
        # Get the attention weight for this timestep
        let a = att[t]
        # Accumulate the weighted value into xb
        for i in 0..<headSize:
          xb[i] = xb[i] + a * v[i]

    # Final matmul to get the output of the attention
    matmul(s.xb2, s.xb, w.wo + l * dim * dim, dim, dim)

    # Residual connection back into x
    for i in 0..<dim:
      x[i] = x[i] + s.xb2[i]

    # FFN rmsnorm
    rmsnorm(s.xb, x, w.rms_ffn_weight + l * dim, dim)

    # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    # First calculate self.w1(x) and self.w3(x)
    matmul(s.hb, s.xb, w.w1 + l * dim * hidden_dim, dim, hidden_dim)
    matmul(s.hb2, s.xb, w.w3 + l * dim * hidden_dim, dim, hidden_dim)

    # SwiGLU non-linearity
    for i in 0..<hidden_dim:
        var val = s.hb[i]
        # silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0 / (1.0 + exp(-val)))
        # Elementwise multiply with w3(x)
        val *= s.hb2[i]
        s.hb[i] = val

    # Final matmul to get the output of the ffn
    matmul(s.xb, s.hb, w.w2 + l * dim * hidden_dim, hidden_dim, dim)

    # Residual connection
    for i in 0..<dim:
      x[i] = x[i] + s.xb[i]

  # Final rmsnorm
  rmsnorm(x, x, w.rms_final_weight, dim)

  # Classifier into logits
  matmul(s.logits, x, w.wcls, p.dim, p.vocab_size)

  return s.logits

proc newSampler(sampler: var Sampler, vocab_size: cint, temperature: float32, topp: float32, rng_seed: uint64) =
  sampler.vocab_size = vocab_size
  sampler.temperature = temperature
  sampler.topp = topp
  sampler.rng_state = rng_seed
  # buffer only used with nucleus sampling; may not need but it's ~small
  sampler.probindex = cast[ptr[ProbIndex]](alloc0(sampler.vocab_size * sizeof(ProbIndex)))

proc sample_argmax(probabilities: ptr float32, n: cint): cint =
  # return the index that has the highest probability
  var
    max_i: cint = 0
    max_p: float32 = probabilities[0]

  for i in 1..<n:
    if probabilities[i] > max_p:
      max_i = i
      max_p = probabilities[i]

  return max_i

proc sample_mult(probabilities: ptr float32, n: cint, coin: float32): cint =
  # Sample index from probabilities (they must sum to 1!)
  # Coin is a random number in [0, 1), usually from random_f32()
  var cdf = 0.0'f32
  for i in 0..<n:
    cdf += probabilities[i]
    if coin < cdf:
      return i
  return n - 1  # In case of rounding errors

proc random_u32(statePtr: ptr uint64): uint32 =
  # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  var state = statePtr[]
  state = state xor (state shr 12)
  state = state xor (state shl 25)
  state = state xor (state shr 27)
  result = ((state * 0x2545F4914F6CDD1D'u64) shr 32).uint32
  statePtr[] = state

proc random_f32(state: ptr uint64): float32 =  # random float32 in [0,1)
  result = (random_u32(state) shr 8).float32 / 16777216.0'f32

proc compare(a, b: ProbIndex): int =
  if a.prob > b.prob:
    return -1
  if a.prob < b.prob:
    return 1
  return 0

proc sample_topp(probabilities: ptr float32, n: cint, topp: float32, probindex: ptr ProbIndex, coin: float32): cint =
  # top-p sampling (or "nucleus sampling") samples from the smallest set of
  # tokens that exceed probability topp. This way we never sample tokens that
  # have very low probabilities and are less likely to go "off the rails".
  # coin is a random number in [0, 1), usually from random_f32()

  var n0: cint = 0
  # quicksort indices in descending order of probabilities
  # values smaller than (1 - topp) / (n - 1) cannot be part of the result
  # so for efficiency we crop these out as candidates before sorting
  let cutoff = (1.0'f32 - topp) / float32(n - 1)
  for i in 0 ..< n:
    if probabilities[i] >= cutoff:
      var prob = ProbIndex()
      prob.index = i.cint
      prob.prob = probabilities[i]
      probindex[n0] = prob
      # probindex[n0].index = i
      # probindex[n0].prob = probabilities[i]
      inc(n0)

  #qsort(probindex, n0.csizet, sizeof(ProbIndex).csizet, compare)
  var probindexSeq = newSeq[ProbIndex](n0.int)
  for i in 0 ..< n0: probindexSeq[i] = probindex[i]
  sort(probindexSeq, compare)
  for i in 0 ..< n0: probindex[i] = probindexSeq[i]

  # truncate the list where cumulative probability exceeds topp
  var cumulative_prob = 0.0'f32
  var last_idx = n0 - 1  # in case of rounding errors consider all elements
  for i in 0..<n0:
    cumulative_prob += probindex[i].prob
    if cumulative_prob > topp:
      last_idx = i
      break  # we've exceeded topp by including last_idx

  # sample from the truncated list
  let r = coin * cumulative_prob
  var cdf = 0.0'f32
  for i in 0 ..< last_idx + 1:
    cdf += probindex[i].prob
    if r < cdf:
      return probindex[i].index

  return probindex[last_idx].index  # in case of rounding errors

proc sample(sampler: ptr Sampler, logits: ptr float32): cint =
  # sample the token given the logits and some hyperparameters
  var next: cint
  if sampler[].temperature == 0.0'f32:
    # greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler[].vocab_size)
  else:
    # apply the temperature to the logits
    for q in 0..<sampler[].vocab_size:
      logits[q] = logits[q] / sampler[].temperature
    # apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler[].vocab_size)
    # flip a (float) coin (this is our source of entropy for sampling)
    let coin = random_f32(addr sampler[].rng_state)
    # we sample from this distribution to get the next token
    if sampler[].topp <= 0 or sampler[].topp >= 1:
      # simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler[].vocab_size, coin)
    else:
      # top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(
        logits,
        sampler[].vocab_size,
        sampler[].topp,
        sampler[].probindex, coin
      )
  return next

proc generate(transformer: ptr Transformer, tokenizer: ptr Tokenizer, sampler: ptr Sampler, prompt: string, steps: cint): string =

  # var numPromptTokens: cint = 0 # Using Nim naming conventions
  # var promptTokens0 = newSeq[cint](prompt.len + 3)

  # encode(tokenizer, prompt.cstring, 1, 0, promptTokens0[0].addr, addr numPromptTokens) # Assuming encode has been defined or imported to Nim
  # if numPromptTokens < 1:
  #   quit("something is wrong, expected at least 1 prompt token")

  # echo "---"
  # for i in 0 ..< numPromptTokens:
  #   echo ": ", promptTokens0[i]


  #echo "---"
  let promptTokens = encode(tokenizer, prompt)
  for token in promptTokens:
    echo ": ", token

  var
    startTime: float
    next: cint
    token = promptTokens[0] # kick off with the first token in the prompt
    pos: cint = 0     # position in the sequence

  while pos < steps:
    # ... rest of the loop
    # forward the transformer to get logits for the next token

    var logits: ptr float32 = forward(transformer, token, pos)

    # advance the state state machine
    if pos < promptTokens.len - 1:
      # if we are still processing the input prompt, force the next prompt token
      next = promptTokens[pos + 1]
    else:
      # otherwise sample the next token from the logits
      next = sample(sampler, logits)
    pos.inc

    # data-dependent terminating condition: the BOS (=1) token delimits sequences
    if next == 1:
      break

    # print the token as string, decode it with the Tokenizer object
    let piece = decode(tokenizer, token, next)

    stdout.write(piece)
    stdout.flushFile()
    result.add(piece)
    token = next

    # quit("\nok...")

    if startTime == 0:
      startTime = epochTime()

  echo "\n"
  result.add("\n")

  if pos > 1:
    let endTime = epochTime()
    echo "achieved tok/s: ", $(pos.float/(endTime - startTime))

const
  Help  = {
    "model": "Model file",
    "temperature": "temperature in [0,inf], default 1.0",
    "pvalue": "p value in top-p (nucleus) sampling in [0,1] default 0.9",
    "seed": "random seed, default time(NULL)",
    "steps": "number of steps to run for, default 256. 0 = max_seq_len",
    "input": "input prompt",
    "tokenizer": "optional path to custom tokenizer",
    "mode": "mode: generate|chat, default: generate",
    "sysPrompt": "(optional) system prompt in chat mode"
  }.toTable()

  Short = {
    "model": 'm',
    "temperature": 't',
    "pvalue": 'p',
    "seed": 's',
    "steps": 'n',
    "input": 'i',
    "tokenizer": 'z',
    "mode": 'd',
    "sysPrompt": 'y'
  }.toTable()

proc main*(
  model: string,
  temperature: float32 = 1.0,
  pvalue: float32 = 0.9,
  seed: int = int(getTime().toUnix),  # Default to current Unix time
  steps: int = 256,
  input: string = "",
  tokenizer: string = "tokenizer.bin",
  mode: string = "generate",
  sysPrompt: string = "Only short answers"
) =

  var
    checkpoint_path = model
    tokenizer_path = tokenizer
    temperature = temperature
    topp = pvalue
    steps = steps
    prompt = input
    rng_seed = seed
    mode = mode
    system_prompt = sysPrompt

  if rng_seed <= 0:
    rng_seed = int(getTime().toUnix)
  if temperature < 0.0:
    temperature = 0.0
  if topp < 0.0 or 1.0 < topp:
    topp = 0.9
  if steps < 0:
    steps = 0

  echo "build the Transformer via the model .bin file"
  var transformer: Transformer
  newTransformer(transformer, checkpoint_path)

  echo "build the Tokenizer via the tokenizer .bin file"
  var tokenizer: Tokenizer
  #build_tokenizer(tokenizer.addr, tokenizer_path.cstring, transformer.config.vocab_size)
  newTokenizer(tokenizer, tokenizer_path, transformer.config.vocab_size)

  echo "build the Sampler"
  var sampler: Sampler
  newSampler(sampler, transformer.config.vocab_size, temperature, topp, rng_seed.uint64)

  echo "done loading"

  if mode == "generate":
    echo generate(
      transformer.addr,
      tokenizer.addr,
      sampler.addr,
      prompt,
      steps.cint
    )
  # elif mode == "chat":
  #   chat(
  #     transformer.addr,
  #     tokenizer.addr,
  #     sampler.addr,
  #     nil,
  #     system_prompt.cstring,
  #     steps.cint
  #   )
  else:
    quit("Use a valid mode")


when isMainModule:
  dispatch(main, help = Help, short = Short)
