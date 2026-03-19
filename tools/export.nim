import pth, std/streams, std/json, std/strutils, std/strformat,
    ../src/llama2, std/parseopt


# float decode(uint16_t float16_value)
# {
#   // MSB -> LSB
#   // float16=1bit: sign, 5bit: exponent, 10bit: fraction
#   // float32=1bit: sign, 8bit: exponent, 23bit: fraction
#   // for normal exponent(1 to 0x1e): value=2**(exponent-15)*(1.fraction)
#   // for denormalized exponent(0): value=2**-14*(0.fraction)
#   uint32_t sign = float16_value >> 15;
#   uint32_t exponent = (float16_value >> 10) & 0x1F;
#   uint32_t fraction = (float16_value & 0x3FF);
#   uint32_t float32_value;
#   if (exponent == 0)
#   {
#     if (fraction == 0)
#     {
#       // zero
#       float32_value = (sign << 31);
#     }
#     else
#     {
#       // can be represented as ordinary value in float32
#       // 2 ** -14 * 0.0101
#       // => 2 ** -16 * 1.0100
#       // int int_exponent = -14;
#       exponent = 127 - 14;
#       while ((fraction & (1 << 10)) == 0)
#       {
#         //int_exponent--;
#         exponent--;
#         fraction <<= 1;
#       }
#       fraction &= 0x3FF;
#       // int_exponent += 127;
#       float32_value = (sign << 31) | (exponent << 23) | (fraction << 13);
#     }
#   }
#   else if (exponent == 0x1F)
#   {
#     /* Inf or NaN */
#     float32_value = (sign << 31) | (0xFF << 23) | (fraction << 13);
#   }
#   else
#   {
#     /* ordinary number */
#     float32_value = (sign << 31) | ((exponent + (127-15)) << 23) | (fraction << 13);
#   }

#   return *((float*)&float32_value);
# }

type float16 = int16

proc float16ToFloat32(value: float16): float32 =
  ## Converts a float16 to a float32.

  let
    x = cast[uint16](value)
    sign = x shr 15
    exponent = (x shr 10) and 0x1F
    fraction = x and 0x3FF
  var float32Value: uint32

  if exponent == 0:
    if fraction == 0:
      # zero
      float32Value = (sign shl 31).uint32
    else:
      # can be represented as ordinary value in float32
      # 2 ** -14 * 0.0101
      # => 2 ** -16 * 1.0100
      var exp = 127'u32 - 14'u32
      var frac = fraction
      while (frac and (1 shl 10)) == 0:
        dec exp
        frac = frac shl 1
      frac = frac and 0x3FF
      float32Value = ((sign shl 31) or (exp shl 23) or (frac shl 13)).uint32

  elif exponent == 0x1F:
    # Inf or NaN
    float32Value = ((sign shl 31) or (0xFF'u32 shl 23) or (fraction shl 13)).uint32

  else:
    # ordinary number
    float32Value = ((sign shl 31) or ((exponent + (127-15)) shl 23) or (fraction shl 13)).uint32

  return cast[float32](float32Value)

proc loadAndExport(path: string, output: string, verbose = false) =
  #var params = readFile(path & "/params.json").parseJson()

  let torchData = loadTorchData(path)
  defer:
    torchData.close()

  let c = Config(
    dim: 4096,
    hiddenDim: 11008,
    numLayers: 32,
    numHeads: 32,
    numKVHeads: 32,
    vocabSize: -32000,
    seqLen: 2048
  )

  if verbose:
    proc show(name, name2: string) =
      let tensor = torchData.find(name)
      #let loc = reader.getPointer("consolidated/data/" & tensor.storage.fileName)
      #echo hexPrint(loc + (-0x68), 0x100)
      for i in 0 ..< 3:
        let x = cast[ptr[int16]](tensor.data + i*2)[]
        let x1 = float16ToFloat32(x.float16)
        echo &"{name2}[{i}] == {x1:0.6f}"

    show("tok_embeddings.weight", "t.weights.tokenEmbeddingTable")
    show("layers.0.attention_norm.weight", "t.weights.rmsAttWeight")

    show("layers.0.attention.wq.weight", "t.weights.wq")
    show("layers.0.attention.wk.weight", "t.weights.wk")
    show("layers.0.attention.wv.weight", "t.weights.wv")
    show("layers.0.attention.wo.weight", "t.weights.wo")
    show("layers.0.ffn_norm.weight", "t.weights.rmsFfnWeight")
    show("layers.0.feed_forward.w1.weight", "t.weights.w1")
    show("layers.0.feed_forward.w2.weight", "t.weights.w2")
    show("layers.0.feed_forward.w3.weight", "t.weights.w3")

    show("norm.weight", "t.weights.rmsFinalWeight")
    show("output.weight", "t.weights.wcls")

  let f = newFileStream(output, fmWrite)
  proc serialize(f: Stream, name: string) =
    if verbose:
      echo " * ", name
    let tensor = torchData.find(name)
    for i in 0 ..< tensor.dataSize:
      let x = cast[ptr[int16]](tensor.data + i*2)[]
      let x1 = float16ToFloat32(x.float16)
      f.write(x1)

  if verbose:
    echo "writing: " & output
  f.write(c)
  f.serialize("tok_embeddings.weight")
  # now all the layers
  # attention weights
  for i in 0 ..< c.numLayers: f.serialize(&"layers.{i}.attention_norm.weight")
  for i in 0 ..< c.numLayers: f.serialize(&"layers.{i}.attention.wq.weight")
  for i in 0 ..< c.numLayers: f.serialize(&"layers.{i}.attention.wk.weight")
  for i in 0 ..< c.numLayers: f.serialize(&"layers.{i}.attention.wv.weight")
  for i in 0 ..< c.numLayers: f.serialize(&"layers.{i}.attention.wo.weight")
  # ffn weights
  for i in 0 ..< c.numLayers: f.serialize(&"layers.{i}.ffn_norm.weight")
  for i in 0 ..< c.numLayers: f.serialize(&"layers.{i}.feed_forward.w1.weight")
  for i in 0 ..< c.numLayers: f.serialize(&"layers.{i}.feed_forward.w2.weight")
  for i in 0 ..< c.numLayers: f.serialize(&"layers.{i}.feed_forward.w3.weight")

  f.serialize("norm.weight")

  # write unused data
  # f.serialize("freqs_cos")
  for i in 0 ..< c.seqLen * (c.dim div c.numHeads) div 2:
    f.write(0.float32)
  # f.serialize("freqs_sin")
  for i in 0 ..< c.seqLen * (c.dim div c.numHeads) div 2:
    f.write(0.float32)

  f.serialize("output.weight")

  f.close()

when isMainModule:
  # Default values
  var
    inputPath = ""
    outputFile = ""
    verbose = false

  for kind, key, val in getopt():
    case kind
    of cmdArgument:
      discard  # ignore non-option arguments for now
    of cmdLongOption, cmdShortOption:
      case key
      of "i", "input":
        inputPath = val
      of "o", "output":
        outputFile = val
      of "v", "verbose":
        verbose = val == "true"
      else:
        echo "Unknown option: ", key
        quit(1)
    of cmdEnd:
      break

  if inputPath.len == 0:
    quit("--input Model path is required.")
  if outputFile.len == 0:
    quit("--output Model is file required.")

  loadAndExport(inputPath, outputFile, verbose)
