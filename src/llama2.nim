import cligen, std/tables, std/times, math, std/algorithm, std/strutils,
    std/streams, std/os, std/memfiles

{.passC: "-Ofast -funsafe-math-optimizations -ffast-math -mtune=native -march=native".}

type
  Config = object
    dim: cint           # transformer dimension
    hiddenDim: cint    # for ffn layers
    numLayers: cint      # number of layers
    numHeads: cint       # number of query heads
    numKVHeads: cint    # number of key/value heads (can be < query heads because of multiquery)
    vocabSize: cint    # vocabulary size, usually 256 (byte-level)
    seq_len: cint       # max sequence length

  TransformerWeights = object
    # token embedding table
    token_embedding_table: ptr[float32]    # (vocabSize, dim)
    # weights for rmsNorms
    rms_att_weight: ptr[float32] # (layer, dim) rmsNorm weights
    rms_ffn_weight: ptr[float32] # (layer, dim)
    # weights for matMuls. note dim == numHeads * head_size
    wq: ptr[float32] # (layer, dim, numHeads * head_size)
    wk: ptr[float32] # (layer, dim, numKVHeads * head_size)
    wv: ptr[float32] # (layer, dim, numKVHeads * head_size)
    wo: ptr[float32] # (layer, numHeads * head_size, dim)
    # weights for ffn
    w1: ptr[float32] # (layer, hiddenDim, dim)
    w2: ptr[float32] # (layer, dim, hiddenDim)
    w3: ptr[float32] # (layer, hiddenDim, dim)
    # final rmsNorm
    rms_final_weight: ptr[float32] # (dim,)
    # (optional) classifier weights for the logits, on the last layer
    wcls: ptr[float32]

  RunState = object
    # current wave of activations
    x: ptr[float32]   # activation at current time stamp (dim,)
    xb: ptr[float32]  # same, but inside a residual branch (dim,)
    xb2: ptr[float32] # an additional buffer just for convenience (dim,)
    hb: ptr[float32]  # buffer for hidden dimension in the ffn (hiddenDim,)
    hb2: ptr[float32] # buffer for hidden dimension in the ffn (hiddenDim,)
    q: ptr[float32]   # query (dim,)
    k: ptr[float32]   # key (dim,)
    v: ptr[float32]   # value (dim,)
    att: ptr[float32] # buffer for scores/attention values (numHeads, seq_len)
    logits: ptr[float32] # output logits
    # kv cache
    key_cache: ptr[float32]   # (layer, seq_len, dim)
    value_cache: ptr[float32] # (layer, seq_len, dim)

  Transformer = ref object
    config: Config              # the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights # the weights of the model
    state: RunState             # buffers for the "wave" of activations in the forward pass
    # some more state needed to properly clean up the memory mapping (sigh)
    fileSize: int              # size of the checkpoint file in bytes

type
  TokenIndex = object
    str: cstring  # Equivalent of char* in Nim
    id: cint

  Tokenizer = ref object
    vocab: ptr[ptr[char]] # Equivalent of char** in Nim
    vocab_scores: ptr[float32]
    sorted_vocab: ptr[TokenIndex]
    vocabSize: cint
    max_token_length: uint32      # Equivalent of unsigned int in Nim
    byte_pieces: array[512, uint8] # Equivalent of unsigned char[512] in Nim

type
  ProbIndex = object
    prob: float32  # float in C
    index: cint    # int in C

  Sampler = ref object
    vocabSize: cint
    probIndex: ptr[ProbIndex] # equivalent of ProbIndex* in C
    temperature: float32
    topp: float32
    rngState: uint64   # equivalent of unsigned long long in C

proc `+`[T](p: ptr[T], n: SomeInteger): ptr[T] =
  cast[ptr[T]](cast[uint64](p) + n.uint64 * sizeof(T).uint64)

proc `[]`[T](p: ptr[T], n: SomeInteger): T =
  let p2 = p + n
  return p2[]

proc `[]=`[T](p: ptr[T], n: SomeInteger, v: T) =
  let p2 = p + n
  p2[] = v

proc `+`(p: pointer, n: SomeInteger): pointer =
  cast[pointer](cast[uint64](p) + n.uint64)

proc newTransformer(checkpointPath: string): Transformer =
  let t = Transformer()
  var f = memfiles.open(checkpointPath)

  var filePosition = f.mem

  proc readObj[T](obj: var T) =
    copyMem(obj.addr, filePosition, sizeof(obj))
    filePosition = filePosition + sizeof(obj)

  proc readFloats(count: int): ptr[float32] =
    result = cast[ptr[float32]](filePosition)
    filePosition = filePosition + (count * 4)

  var c: Config
  readObj(c)

  let shared_weights =
    if c.vocabSize > 0:
      true
    else:
      false
  c.vocabSize = abs(c.vocabSize)
  t.config = c

  t.fileSize = f.size

  t.weights.token_embedding_table = readFloats(c.vocabSize * c.dim)
  t.weights.rms_att_weight = readFloats(c.numLayers * c.dim)
  t.weights.wq = readFloats(c.numLayers * c.dim * c.dim)
  t.weights.wk = readFloats(c.numLayers * c.dim * c.dim)
  t.weights.wv = readFloats(c.numLayers * c.dim * c.dim)
  t.weights.wo = readFloats(c.numLayers * c.dim * c.dim)
  t.weights.rms_ffn_weight = readFloats(c.numLayers * c.dim)
  t.weights.w1 = readFloats(c.numLayers * c.dim * c.hiddenDim)
  t.weights.w2 = readFloats(c.numLayers * c.hiddenDim * c.dim)
  t.weights.w3 = readFloats(c.numLayers * c.dim * c.hiddenDim)
  t.weights.rms_final_weight = readFloats(c.dim)
  # Skipping unused t.weights.freq_cis_real
  discard readFloats(c.seq_len * (c.dim div c.numHeads) div 2)
  # Skipping unused t.weights.freq_cis_imag =
  discard readFloats(c.seq_len * (c.dim div c.numHeads) div 2)
  if shared_weights:
    t.weights.wcls = t.weights.token_embedding_table
  else:
    # The rest of the file
    t.weights.wcls = cast[ptr float32](filePosition)

  # Allocate run state
  let kvDim = (c.dim * c.numKVHeads) div c.numHeads
  t.state.x = cast[ptr float32](alloc0(c.dim * sizeof(float32)))
  t.state.xb = cast[ptr float32](alloc0(c.dim * sizeof(float32)))
  t.state.xb2 = cast[ptr float32](alloc0(c.dim * sizeof(float32)))
  t.state.hb = cast[ptr float32](alloc0(c.hiddenDim * sizeof(float32)))
  t.state.hb2 = cast[ptr float32](alloc0(c.hiddenDim * sizeof(float32)))
  t.state.q = cast[ptr float32](alloc0(c.dim * sizeof(float32)))
  t.state.k = cast[ptr float32](alloc0(kvDim * sizeof(float32)))
  t.state.v = cast[ptr float32](alloc0(kvDim * sizeof(float32)))
  t.state.att = cast[ptr float32](alloc0(c.numHeads * c.seq_len * sizeof(float32)))
  t.state.logits = cast[ptr float32](alloc0(c.vocabSize * sizeof(float32)))
  t.state.key_cache = cast[ptr float32](alloc0(c.numLayers * c.seq_len * kvDim * sizeof(float32)))
  t.state.value_cache = cast[ptr float32](alloc0(c.numLayers * c.seq_len * kvDim * sizeof(float32)))

  return t

proc newTokenizer(tokenizerPath: string, vocabSize: cint): Tokenizer =
  ## Creates a new tokenizer given path to a bin.
  let t = Tokenizer()
  let f = newFileStream(tokenizerPath)
  t.vocabSize = vocabSize

  t.vocab = cast[ptr ptr char](alloc0(vocabSize * sizeof(ptr char)))
  t.vocab_scores = cast[ptr float32](alloc0(vocabSize * sizeof(float32)))

  for i in 0..255:
    t.byte_pieces[i * 2] = i.uint8
    t.byte_pieces[i * 2 + 1] = 0.uint8

  t.max_token_length = f.readUInt32()
  for i in 0 ..< vocabSize:
    t.vocab_scores[i] = f.readFloat32()
    let len = f.readInt32()
    let bstr = f.readStr(len)
    t.vocab[i] = cast[ptr char](alloc0(len+1))
    for j in 0 ..< len:
      t.vocab[i][j] = bstr[j]

  f.close()

  return t

proc decode(t: Tokenizer, prev_token: cint, token: cint): string =
  ## Takes a tokenID and turns it into a string part.
  var piece = t.vocab[token]
  for i in 0 .. 32:
    if piece[i] == '\0':
      break
    result.add piece[i]

  if prev_token == 1 and result[0] == ' ':
    result = result[1 .. ^1]

  if result == "<0x0A>":
    result = "\n"

proc getToken(t: Tokenizer, tokenId: cint): string =
  ## Given a tokenId returns its string part.
  var token = ""
  var piece = t.vocab[tokenId]
  for i in 0 .. 32:
    if piece[i] == '\0':
      break
    token.add piece[i]
  return token

proc findToken(t: Tokenizer, stringPart: string): cint =
  ## Given a token string part, returns its tokenId.
  for tokenId in 0 ..< t.vocabSize:
    var token = t.getToken(tokenId)
    if token == stringPart:
      return tokenId.cint
  return -1

proc encode(t: Tokenizer, text: string): seq[cint] =
  ## Takes a string and encodes it into seq of token Ids.

  var tokens: seq[cint]
  tokens.add 1

  if text == "":
    return tokens

  var text = " " & text

  # First encode every individual character in the input text
  for c in text:
    let tokenId = t.findToken($c)
    if tokenId == -1:
      quit("Error encoding")
    tokens.add(tokenId.cint)

  # Merge the best consecutive pair each iteration, according to the scores in vocab_scores
  while true:
    var
      bestScore = float32.low
      bestId = -1.cint
      bestIdx = -1.cint

    for i in 0 ..< tokens.len - 1:
      # Check if we can merge the pair (tokens[i], tokens[i+1])
      var str = t.getToken(tokens[i]) & t.getToken(tokens[i + 1])
      let tokenId = t.findToken(str)
      if tokenId != -1 and t.vocab_scores[tokenId] > bestScore:
        # This merge pair exists in vocab! Record its score and position
        bestScore = t.vocab_scores[tokenId]
        bestId = tokenId
        bestIdx = i.cint

    if bestIdx == -1:
      break  # We couldn't find any more pairs to merge, so we're done

    # Merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
    tokens[bestIdx] = bestId
    # Delete token at position bestIdx+1, shift the entire sequence back 1
    tokens.delete(bestIdx + 1)

  return tokens

proc rmsNorm(o: ptr float32, x: ptr float32, weight: ptr float32, size: cint) =
  # Calculate sum of squares
  var ss = 0.0'f32
  for j in 0 ..< size:
    ss += x[j] * x[j]
  ss /= size.float32
  ss += 1e-5'f32
  ss = 1.0'f32 / sqrt(ss)
  # Normalize and scale
  for j in 0 ..< size:
    o[j] = weight[j] * (ss * x[j])

proc softMax(x: ptr float32, size: cint) =
  # Find max value (for numerical stability)
  var maxVal = x[0]
  for i in 1 ..< size:
    if x[i] > maxVal:
      maxVal = x[i]

  # exp and sum
  var sum = 0.0'f32
  for i in 0 ..< size:
    x[i] = exp(x[i] - maxVal)
    sum += x[i]

  # Normalize
  for i in 0 ..< size:
    x[i] = x[i] / sum

proc matMul(xout: ptr float32, x: ptr float32, w: ptr float32, n: cint, d: cint) =
  # W (d,n) @ x (n,) -> xout (d,)
  # by far the most amount of time is spent inside this little function

  #echo "matMul ", $n, "x", $d

  for i in 0 ..< d:
    var val = 0.0'f32
    for j in 0 ..< n:
      val += w[i * n + j] * x[j]
    xout[i] = val

proc forward(transformer: Transformer, token: cint, pos: cint): ptr float32 =
  # Convenience variables
  let p = addr transformer.config
  let w = addr transformer.weights
  let s = addr transformer.state
  let x = s.x
  let dim = p.dim
  let kvDim = (p.dim * p.numKVHeads) div p.numHeads
  let kvMul = p.numHeads div p.numKVHeads  # Integer multiplier of the kv sharing in multiquery
  let hiddenDim = p.hiddenDim
  let head_size = dim div p.numHeads

  # Copy the token embedding into x
  let contentRow = w.token_embedding_table + token * dim
  copyMem(x, contentRow, dim * sizeof(x[]))

  # Forward all the layers
  for l in 0 ..< p.numLayers:

    # Attention rmsNorm
    rmsNorm(s.xb, x, w.rms_att_weight + l*dim, dim)

    # QKV matMuls for this position
    matMul(s.q, s.xb, w.wq + l*dim*dim, dim, dim)
    matMul(s.k, s.xb, w.wk + l*dim*kvDim, dim, kvDim)
    matMul(s.v, s.xb, w.wv + l*dim*kvDim, dim, kvDim)

    # RoPE relative positional encoding: complex-valued rotate q and k in each head
    for i in countup(0, dim - 1, 2):
      let headDim = i mod headSize
      let freq = 1.0'f32 / pow(10000.0'f32, float32(headDim) / float32(headSize))
      let val = float32(pos) * freq
      let fcr = cos(val)
      let fci = sin(val)
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

    # Multi-head attention. Iterate over all heads
    # var h: cint
    # pragma omp parallel for private(h)
    for h in 0 ..< p.numHeads:
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
        for i in 0 ..< headSize:
            score += q[i] * k[i]
        score /= sqrt(float32(headSize))
        # Save the score to the attention buffer
        att[t] = score

      # SoftMax the scores to get attention weights, from 0..pos inclusively
      softMax(att, pos + 1)

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

    # Final matMul to get the output of the attention
    matMul(s.xb2, s.xb, w.wo + l * dim * dim, dim, dim)

    # Residual connection back into x
    for i in 0..<dim:
      x[i] = x[i] + s.xb2[i]

    # FFN rmsNorm
    rmsNorm(s.xb, x, w.rms_ffn_weight + l * dim, dim)

    # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    # First calculate self.w1(x) and self.w3(x)
    matMul(s.hb, s.xb, w.w1 + l * dim * hiddenDim, dim, hiddenDim)
    matMul(s.hb2, s.xb, w.w3 + l * dim * hiddenDim, dim, hiddenDim)

    # SwiGLU non-linearity
    for i in 0..<hiddenDim:
      var val = s.hb[i]
      # silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0 / (1.0 + exp(-val)))
      # Elementwise multiply with w3(x)
      val *= s.hb2[i]
      s.hb[i] = val

    # Final matMul to get the output of the ffn
    matMul(s.xb, s.hb, w.w2 + l * dim * hiddenDim, hiddenDim, dim)

    # Residual connection
    for i in 0..<dim:
      x[i] = x[i] + s.xb[i]

  # Final rmsNorm
  rmsNorm(x, x, w.rms_final_weight, dim)

  # Classifier into logits
  matMul(s.logits, x, w.wcls, p.dim, p.vocabSize)

  return s.logits

proc newSampler(vocabSize: cint, temperature: float32, topp: float32, rngSeed: uint64): Sampler =
  let sampler = Sampler()
  sampler.vocabSize = vocabSize
  sampler.temperature = temperature
  sampler.topp = topp
  sampler.rngState = rngSeed
  # buffer only used with nucleus sampling; may not need but it's ~small
  sampler.probIndex = cast[ptr[ProbIndex]](alloc0(sampler.vocabSize * sizeof(ProbIndex)))
  return sampler

proc sampleArgmax(probabilities: ptr float32, n: cint): cint =
  # return the index that has the highest probability
  var
    maxI: cint = 0
    maxP: float32 = probabilities[0]

  for i in 1 ..< n:
    if probabilities[i] > maxP:
      maxI = i
      maxP = probabilities[i]

  return maxI

proc sampleMult(probabilities: ptr float32, n: cint, coin: float32): cint =
  # Sample index from probabilities (they must sum to 1!)
  # Coin is a random number in [0, 1), usually from randomFloat32()
  var cdf = 0.0'f32
  for i in 0 ..< n:
    cdf += probabilities[i]
    if coin < cdf:
      return i
  return n - 1  # In case of rounding errors

proc randomUInt32(statePtr: ptr uint64): uint32 =
  ## Compute random uint32
  # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  var state = statePtr[]
  state = state xor (state shr 12)
  state = state xor (state shl 25)
  state = state xor (state shr 27)
  result = ((state * 0x2545F4914F6CDD1D'u64) shr 32).uint32
  statePtr[] = state

proc randomFloat32(state: ptr uint64): float32 =
  ## Compute andom float32 in [0,1)
  result = (randomUInt32(state) shr 8).float32 / 16777216.0'f32

proc compare(a, b: ProbIndex): int =
  ## Compare two prob indexes for sorting.
  if a.prob > b.prob:
    return -1
  if a.prob < b.prob:
    return 1
  return 0

proc sampleTopP(probabilities: ptr float32, n: cint, topp: float32, probIndex: ptr ProbIndex, coin: float32): cint =
  # top-p sampling (or "nucleus sampling") samples from the smallest set of
  # tokens that exceed probability topp. This way we never sample tokens that
  # have very low probabilities and are less likely to go "off the rails".
  # coin is a random number in [0, 1), usually from randomFloat32()

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
      probIndex[n0] = prob
      inc(n0)

  # TODO: remove this Copy
  var probIndexSeq = newSeq[ProbIndex](n0.int)
  for i in 0 ..< n0: probIndexSeq[i] = probIndex[i]
  sort(probIndexSeq, compare)
  for i in 0 ..< n0: probIndex[i] = probIndexSeq[i]

  # truncate the list where cumulative probability exceeds topp
  var cumulative_prob = 0.0'f32
  var last_idx = n0 - 1  # in case of rounding errors consider all elements
  for i in 0..<n0:
    cumulative_prob += probIndex[i].prob
    if cumulative_prob > topp:
      last_idx = i
      break  # we've exceeded topp by including last_idx

  # sample from the truncated list
  let r = coin * cumulative_prob
  var cdf = 0.0'f32
  for i in 0 ..< last_idx + 1:
    cdf += probIndex[i].prob
    if r < cdf:
      return probIndex[i].index

  return probIndex[last_idx].index  # in case of rounding errors

proc sample(sampler: Sampler, logits: ptr float32): cint =
  # Sample the token given the logits and some hyperparameters
  var next: cint
  if sampler.temperature == 0.0'f32:
    # Greedy argmax sampling: take the token with the highest probability
    next = sampleArgmax(logits, sampler.vocabSize)
  else:
    # Apply the temperature to the logits
    for q in 0 ..< sampler.vocabSize:
      logits[q] = logits[q] / sampler.temperature
    # Apply softMax to the logits to get the probabilities for next token
    softMax(logits, sampler.vocabSize)
    # Flip a (float) coin (this is our source of entropy for sampling)
    let coin = randomFloat32(unsafeAddr sampler.rngState)
    # We sample from this distribution to get the next token
    if sampler.topp <= 0 or sampler.topp >= 1:
      # Simply sample from the predicted probability distribution
      next = sampleMult(logits, sampler.vocabSize, coin)
    else:
      # Top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sampleTopP(
        logits,
        sampler.vocabSize,
        sampler.topp,
        sampler.probIndex, coin
      )
  return next

proc generate(transformer: Transformer, tokenizer: Tokenizer, sampler: Sampler, prompt: string, steps: cint, interactive = true): string =
  ## Given a loaded model, generates the output.

  let promptTokens = encode(tokenizer, prompt)

  var
    startTime: float
    next: cint
    token = promptTokens[0] # kick off with the first token in the prompt
    pos: cint = 0 # position in the sequence

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

    if piece == "</s>":
      break

    if interactive:
      stdout.write(piece)
      stdout.flushFile()

    result.add(piece)
    token = next

    if startTime == 0:
      startTime = epochTime()

  if interactive:
    echo "\n"
  result.add("\n")

  if pos > 1 and interactive:
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
    rngSeed = seed
    mode = mode
    system_prompt = sysPrompt

  if rngSeed <= 0:
    rngSeed = int(getTime().toUnix)
  if temperature < 0.0:
    temperature = 0.0
  if topp < 0.0 or 1.0 < topp:
    topp = 0.9
  if steps < 0:
    steps = 0

  echo "build the Transformer via the model .bin file"
  var transformer = newTransformer(checkpoint_path)

  echo "build the Tokenizer via the tokenizer .bin file"
  var tokenizer = newTokenizer(tokenizer_path, transformer.config.vocabSize)

  echo "build the Sampler"
  var sampler = newSampler(transformer.config.vocabSize, temperature, topp, rngSeed.uint64)

  echo "done loading"

  if mode == "generate":
    discard generate(
      transformer,
      tokenizer,
      sampler,
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
