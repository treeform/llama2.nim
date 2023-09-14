import pth, std/streams, std/json, std/strutils, numby, std/strformat,
    ../src/llama2, std/parseopt

proc loadAndExport(path: string, output: string, verbose = false) =
  var params = readFile(path & "/params.json").parseJson()

  let torchData = loadTorchData(path)

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
    inputDir = ""
    outputFile = ""
    verbose = false

  for kind, key, val in getopt():
    case kind
    of cmdArgument:
      discard  # ignore non-option arguments for now
    of cmdLongOption, cmdShortOption:
      case key
      of "i", "input":
        inputDir = val
      of "o", "output":
        outputFile = val
      of "v", "verbose":
        verbose = val == "true"
      else:
        echo "Unknown option: ", key
        quit(1)
    of cmdEnd:
      break

  if inputDir.len == 0:
    quit("--input Model is directory required.")
  if outputFile.len == 0:
    quit("--output Model is file required.")

  loadAndExport(inputDir, outputFile, true)
