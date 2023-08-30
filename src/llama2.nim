import std/tables, std/times, math, std/algorithm, std/strutils,
    std/streams, std/memfiles, std/parseopt

{.passC: "-Ofast -funsafe-math-optimizations -ffast-math -mtune=native -march=native".}

type
  Config = object
    dim*: int32          # transformer dimension
    hiddenDim*: int32    # for ffn layers
    numLayers*: int32    # number of layers
    numHeads*: int32     # number of query heads
    numKVHeads*: int32   # number of key/value heads (can be < query heads because of multiquery)
    vocabSize*: int32    # vocabulary size, usually 256 (byte-level)
    seqLen*: int32       # max sequence length

  TransformerWeights = object
    # token embedding table
    tokenEmbeddingTable: ptr[float32]    # (vocabSize, dim)
    # weights for rmsNorms
    rmsAttWeight: ptr[float32] # (layer, dim) rmsNorm weights
    rmsFfnWeight: ptr[float32] # (layer, dim)
    # weights for matMuls. note dim == numHeads * headSize
    wq: ptr[float32] # (layer, dim, numHeads * headSize)
    wk: ptr[float32] # (layer, dim, numKVHeads * headSize)
    wv: ptr[float32] # (layer, dim, numKVHeads * headSize)
    wo: ptr[float32] # (layer, numHeads * headSize, dim)
    # weights for ffn
    w1: ptr[float32] # (layer, hiddenDim, dim)
    w2: ptr[float32] # (layer, dim, hiddenDim)
    w3: ptr[float32] # (layer, hiddenDim, dim)
    # final rmsNorm
    rmsFinalWeight: ptr[float32] # (dim,)
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
    att: ptr[float32] # buffer for scores/attention values (numHeads, seqLen)
    logits: ptr[float32] # output logits
    # kv cache
    keyCache: ptr[float32]   # (layer, seqLen, dim)
    valueCache: ptr[float32] # (layer, seqLen, dim)

  Transformer* = ref object
    config*: Config              # the hyperparameters of the architecture (the blueprint)
    weights*: TransformerWeights # the weights of the model
    state*: RunState             # buffers for the "wave" of activations in the forward pass
    # some more state needed to properly clean up the memory mapping (sigh)
    fileSize*: int              # size of the checkpoint file in bytes

type
  Tokenizer* = ref object
    vocab: seq[string]
    vocabRev: Table[string, int]
    vocabScores: seq[float32]
    vocabSize: int32
    maxTokenLength: uint32      # Equivalent of unsigned int in Nim

type
  ProbIndex = object
    prob: float32  # float in C
    index: int32    # int in C

  Sampler* = ref object
    vocabSize: int32
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

proc newTransformer*(checkpointPath: string): Transformer =
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

  let sharedWeights =
    if c.vocabSize > 0:
      true
    else:
      false
  c.vocabSize = abs(c.vocabSize)
  t.config = c

  t.fileSize = f.size

  t.weights.tokenEmbeddingTable = readFloats(c.vocabSize * c.dim)
  t.weights.rmsAttWeight = readFloats(c.numLayers * c.dim)
  t.weights.wq = readFloats(c.numLayers * c.dim * c.dim)
  t.weights.wk = readFloats(c.numLayers * c.dim * c.dim)
  t.weights.wv = readFloats(c.numLayers * c.dim * c.dim)
  t.weights.wo = readFloats(c.numLayers * c.dim * c.dim)
  t.weights.rmsFfnWeight = readFloats(c.numLayers * c.dim)
  t.weights.w1 = readFloats(c.numLayers * c.dim * c.hiddenDim)
  t.weights.w2 = readFloats(c.numLayers * c.hiddenDim * c.dim)
  t.weights.w3 = readFloats(c.numLayers * c.dim * c.hiddenDim)
  t.weights.rmsFinalWeight = readFloats(c.dim)
  # Skipping unused t.weights.freqCisReal
  discard readFloats(c.seqLen * (c.dim div c.numHeads) div 2)
  # Skipping unused t.weights.freqCisImag =
  discard readFloats(c.seqLen * (c.dim div c.numHeads) div 2)
  if sharedWeights:
    t.weights.wcls = t.weights.tokenEmbeddingTable
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
  t.state.att = cast[ptr float32](alloc0(c.numHeads * c.seqLen * sizeof(float32)))
  t.state.logits = cast[ptr float32](alloc0(c.vocabSize * sizeof(float32)))
  t.state.keyCache = cast[ptr float32](alloc0(c.numLayers * c.seqLen * kvDim * sizeof(float32)))
  t.state.valueCache = cast[ptr float32](alloc0(c.numLayers * c.seqLen * kvDim * sizeof(float32)))

  return t

proc newTokenizer*(tokenizerPath: string, vocabSize: int32): Tokenizer =
  ## Creates a new tokenizer given path to a bin.
  let t = Tokenizer()
  let f = newFileStream(tokenizerPath)
  t.vocabSize = vocabSize
  t.vocab = newSeq[string](vocabSize)
  t.vocabScores = newSeq[float32](vocabSize)
  t.maxTokenLength = f.readUInt32()
  for i in 0 ..< vocabSize:
    t.vocabScores[i] = f.readFloat32()
    let len = f.readInt32()
    t.vocab[i] = f.readStr(len)
    t.vocabRev[t.vocab[i]] = i
  f.close()
  return t

proc decode*(t: Tokenizer, prevToken: int32, token: int32): string =
  ## Takes a tokenID and turns it into a string part.
  var piece = t.vocab[token]
  result.add piece

  if prevToken == 1 and result[0] == ' ':
    result = result[1 .. ^1]

  if result == "<0x0A>":
    result = "\n"

proc encode*(t: Tokenizer, text: string): seq[int32] =
  ## Takes a string and encodes it into seq of token Ids.
  var tokens: seq[int32]
  tokens.add 1

  if text == "":
    return tokens

  var text = " " & text

  # First encode every individual character in the input text
  for c in text:
    let tokenId = t.vocabRev.getOrDefault($c, -1)
    if tokenId == -1:
      quit("Error encoding")
    tokens.add(tokenId.int32)

  # Merge the best consecutive pair each iteration, according to the scores in vocabScores
  while true:
    var
      bestScore = float32.low
      bestId = -1.int32
      bestIdx = -1.int32

    for i in 0 ..< tokens.len - 1:
      # Check if we can merge the pair (tokens[i], tokens[i+1])
      var str = t.vocab[tokens[i]] & t.vocab[tokens[i + 1]]
      let tokenId = t.vocabRev.getOrDefault(str, -1).int32
      if tokenId != -1 and t.vocabScores[tokenId] > bestScore:
        # This merge pair exists in vocab! Record its score and position
        bestScore = t.vocabScores[tokenId]
        bestId = tokenId
        bestIdx = i.int32

    if bestIdx == -1:
      break  # We couldn't find any more pairs to merge, so we're done

    # Merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
    tokens[bestIdx] = bestId
    # Delete token at position bestIdx+1
    tokens.delete(bestIdx + 1)

  return tokens

proc rmsNorm(dest: ptr float32, x: ptr float32, weight: ptr float32, size: int32) =
  ## Calculate sum of squares
  var ss = 0.0'f32
  for j in 0 ..< size:
    ss += x[j] * x[j]
  ss /= size.float32
  ss += 1e-5'f32
  ss = 1.0'f32 / sqrt(ss)
  # Normalize and scale
  for j in 0 ..< size:
    dest[j] = weight[j] * (ss * x[j])

proc softMax(x: ptr float32, size: int32) =
  ## Find max value (for numerical stability)
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

proc matMul(dest: ptr float32, x: ptr float32, w: ptr float32, n: int32, d: int32) =
  ## Matrix vector multiply.
  # W (d,n) @ x (n,) -> xout (d,)
  # by far the most amount of time is spent inside this little function
  for i in 0 ..< d:
    var val = 0.0'f32
    for j in 0 ..< n:
      val += w[i * n + j] * x[j]
    dest[i] = val

proc forward(transformer: Transformer, token: int32, pos: int32): ptr float32 =
  ## Forward pass of the neural network. The actual AI thinking part.

  # Convenience variables
  let p = addr transformer.config
  let w = addr transformer.weights
  let s = addr transformer.state
  let x = s.x
  let dim = p.dim
  let kvDim = (p.dim * p.numKVHeads) div p.numHeads
  let kvMul = p.numHeads div p.numKVHeads  # Integer multiplier of the kv sharing in multiquery
  let hiddenDim = p.hiddenDim
  let headSize = dim div p.numHeads

  # Copy the token embedding into x
  let contentRow = w.tokenEmbeddingTable + token * dim
  copyMem(x, contentRow, dim * sizeof(x[]))

  # Forward all the layers
  for l in 0 ..< p.numLayers:

    # Attention rmsNorm
    rmsNorm(s.xb, x, w.rmsAttWeight + l*dim, dim)

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
    # var h: int32
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
    rmsNorm(s.xb, x, w.rmsFfnWeight + l * dim, dim)

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
  rmsNorm(x, x, w.rmsFinalWeight, dim)

  # Classifier into logits
  matMul(s.logits, x, w.wcls, p.dim, p.vocabSize)

  return s.logits

proc newSampler*(vocabSize: int32, temperature: float32, topp: float32, rngSeed: uint64): Sampler =
  ## Creates new sampler.
  let sampler = Sampler()
  sampler.vocabSize = vocabSize
  sampler.temperature = temperature
  sampler.topp = topp
  sampler.rngState = rngSeed
  # buffer only used with nucleus sampling; may not need but it's ~small
  sampler.probIndex = cast[ptr[ProbIndex]](alloc0(sampler.vocabSize * sizeof(ProbIndex)))
  return sampler

proc sampleArgmax(probabilities: ptr float32, n: int32): int32 =
  ## Returns the index that has the highest probability
  var
    maxI: int32 = 0
    maxP: float32 = probabilities[0]
  for i in 1 ..< n:
    if probabilities[i] > maxP:
      maxI = i
      maxP = probabilities[i]
  return maxI

proc sampleMult(probabilities: ptr float32, n: int32, coin: float32): int32 =
  ## Sample index from probabilities (they must sum to 1!)
  ## Coin is a random number in [0, 1), usually from randomFloat32()
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
  ## Compute random float32 in [0,1)
  result = (randomUInt32(state) shr 8).float32 / 16777216.0'f32

proc compare(a, b: ProbIndex): int =
  ## Compare two prob indexes for sorting.
  if a.prob > b.prob:
    return -1
  if a.prob < b.prob:
    return 1
  return 0

proc sampleTopP(probabilities: ptr float32, n: int32, topp: float32, probIndex: ptr ProbIndex, coin: float32): int32 =
  # top-p sampling (or "nucleus sampling") samples from the smallest set of
  # tokens that exceed probability topp. This way we never sample tokens that
  # have very low probabilities and are less likely to go "off the rails".
  # coin is a random number in [0, 1), usually from randomFloat32()

  var n0: int32 = 0
  # quicksort indices in descending order of probabilities
  # values smaller than (1 - topp) / (n - 1) cannot be part of the result
  # so for efficiency we crop these out as candidates before sorting
  let cutoff = (1.0'f32 - topp) / float32(n - 1)
  for i in 0 ..< n:
    if probabilities[i] >= cutoff:
      var prob = ProbIndex()
      prob.index = i.int32
      prob.prob = probabilities[i]
      probIndex[n0] = prob
      inc(n0)

  # TODO: remove this Copy
  var probIndexSeq = newSeq[ProbIndex](n0.int)
  for i in 0 ..< n0: probIndexSeq[i] = probIndex[i]
  sort(probIndexSeq, compare)
  for i in 0 ..< n0: probIndex[i] = probIndexSeq[i]

  # truncate the list where cumulative probability exceeds topp
  var cumulativeProb = 0.0'f32
  var lastIdx = n0 - 1  # in case of rounding errors consider all elements
  for i in 0..<n0:
    cumulativeProb += probIndex[i].prob
    if cumulativeProb > topp:
      lastIdx = i
      break  # we've exceeded topp by including lastIdx

  # sample from the truncated list
  let r = coin * cumulativeProb
  var cdf = 0.0'f32
  for i in 0 ..< lastIdx + 1:
    cdf += probIndex[i].prob
    if r < cdf:
      return probIndex[i].index

  return probIndex[lastIdx].index  # in case of rounding errors

proc sample(sampler: Sampler, logits: ptr float32): int32 =
  ## Sample the token given the logits and some hyperparameters
  var next: int32
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

proc generate*(transformer: Transformer, tokenizer: Tokenizer, sampler: Sampler, prompt: string, steps: int32, interactive = false): string =
  ## Given a loaded model, generates the output.

  let promptTokens = encode(tokenizer, prompt)

  var
    startTime: float
    next: int32
    token = promptTokens[0] # kick off with the first token in the prompt
    pos: int32 = 0 # position in the sequence

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

    if "</s>" in piece:
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
    "steps": "number of steps to run for, default 256. 0 = maxSeqLen",
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
  ## Main function
  var
    checkpointPath = model
    tokenizerPath = tokenizer
    temperature = temperature
    topp = pvalue
    steps = steps
    prompt = input
    rngSeed = seed
    mode = mode
    systemPrompt = sysPrompt

  if rngSeed <= 0:
    rngSeed = int(getTime().toUnix)
  if temperature < 0.0:
    temperature = 0.0
  if topp < 0.0 or 1.0 < topp:
    topp = 0.9
  if steps < 0:
    steps = 0

  echo "build the Transformer via the model .bin file"
  var transformer = newTransformer(checkpointPath)

  echo "build the Tokenizer via the tokenizer .bin file"
  var tokenizer = newTokenizer(tokenizerPath, transformer.config.vocabSize)

  echo "build the Sampler"
  var sampler = newSampler(transformer.config.vocabSize, temperature, topp, rngSeed.uint64)

  echo "done loading"

  if mode == "generate":
    discard generate(
      transformer,
      tokenizer,
      sampler,
      prompt,
      steps.int32,
      interactive = true
    )
  # elif mode == "chat":
  #   chat(
  #     transformer.addr,
  #     tokenizer.addr,
  #     sampler.addr,
  #     nil,
  #     systemPrompt.cstring,
  #     steps.int32
  #   )
  else:
    quit("Use a valid mode")

when isMainModule:
  # Default values
  var
    model = ""
    temperature = 1.0
    pValue = 0.9
    seed = int(getTime().toUnix)
    steps = 256
    input = ""
    tokenizer = "tokenizer.bin"
    mode = "generate"
    sysPrompt = "Only short answers"

  for kind, key, val in getopt():
    case kind
    of cmdArgument:
      discard  # ignore non-option arguments for now
    of cmdLongOption, cmdShortOption:
      case key
      of "m", "model":
        model = val
      of "t", "temperature":
        temperature = parseFloat(val)
      of "p", "pvalue":
        pValue = parseFloat(val)
      of "s", "seed":
        seed = parseInt(val)
      of "n", "steps":
        steps = parseInt(val)
      of "i", "input":
        input = val
      of "k", "tokenizer":
        tokenizer = val
      of "o", "mode":
        mode = val
      of "y", "sysPrompt":
        sysPrompt = val
      else:
        echo "Unknown option: ", key
        quit(1)
    of cmdEnd:
      break

  if model.len == 0:
    quit("Model is required.")

  main(model, temperature, pValue, seed, steps, input, tokenizer, mode, sysPrompt)
