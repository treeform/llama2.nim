import pickles, pth, std/streams, std/json, std/os, std/memfiles,
    flatty/hexprint, std/strutils, numby, std/strformat, ../src/llama2,
    zippy/ziparchives

proc `+`(p: pointer, n: SomeInteger): pointer =
  cast[pointer](cast[uint64](p) + n.uint64)

proc `[]`(p: pointer, n: SomeInteger): uint8 =
  return cast[ptr[uint8]](p + n)[]

proc readStr(f: MemFile, offset, size: int): string =
  result = newString(size)
  copyMem(result[0].addr, f.mem + offset, size)

proc readInt16(f: MemFile, offset: int): int16 =
  copyMem(result.addr, f.mem + offset, 2)

proc hexPrint(p: pointer, size: int): string =
  var s = newString(size)
  copyMem(cast[pointer](s[0].addr), p, size)
  hexPrint(s)

proc get(tensors: seq[Tensor], name: string): Tensor =
  for tensor in tensors:
    if tensor.name == name:
      return tensor

proc loadAndExport(path: string, output: string) =
  echo readFile(path & "/params.json").parseJson()

  for i in 0 ..< 10:
    let pthPath = path & "/consolidated.0" & $i & ".pth"
    if fileExists(pthPath):
      echo pthPath

      let reader = openZipArchive(pthPath)
      # Iterate over the paths in the zip archive.
      for path in reader.walkFiles:
        echo path

      doAssert reader.extractFile("consolidated/version").strip() == "3"

      let
        dataPickle = reader.extractFile("consolidated/data.pkl")
        dataJson = dataPickle.pickleToJson(false)
      var tensors = toTensors(dataJson)

      let c = Config(
        dim: 4096,
        hiddenDim: 11008,
        numLayers: 32,
        numHeads: 32,
        numKVHeads: 32,
        vocabSize: -32000,
        seqLen: 2048
      )

      for i, tensor in tensors:
        echo tensor[]
        echo reader.contains("consolidated/data/" & tensor.storage.fileName)

      proc show(name, name2: string) =
        let tensor = tensors.get(name)
        let loc = reader.getPointer("consolidated/data/" & tensor.storage.fileName)
        #echo hexPrint(loc + (-0x68), 0x100)
        for i in 0 ..< 3:
          let x = cast[ptr[int16]](loc + i*2)[]
          let x1 = float16ToFloat32(x.float16)
          echo &"{name2}[{i}] == {x1:0.6f}"

      # # tokenEmbeddingTable

      # # tokenEmbeddingTable (vocabSize, dim)
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

      let f2 = newFileStream("output.m", fmWrite)

      proc writeTensor(f2: Stream, name: string) =
        echo " * ", name
        let tensor = tensors.get(name)
        let loc = reader.getPointer("consolidated/data/" & tensor.storage.fileName)
        for i in 0 ..< tensor.storageSize:
          let x = cast[ptr[int16]](loc + i*2)[]
          let x1 = float16ToFloat32(x.float16)
          f2.write(x1)

      echo "writing: "
      f2.write(c)
      f2.writeTensor("tok_embeddings.weight")
      # now all the layers
      # attention weights
      for i in 0 ..< c.numLayers: f2.writeTensor(&"layers.{i}.attention_norm.weight")
      for i in 0 ..< c.numLayers: f2.writeTensor(&"layers.{i}.attention.wq.weight")
      for i in 0 ..< c.numLayers: f2.writeTensor(&"layers.{i}.attention.wk.weight")
      for i in 0 ..< c.numLayers: f2.writeTensor(&"layers.{i}.attention.wv.weight")
      for i in 0 ..< c.numLayers: f2.writeTensor(&"layers.{i}.attention.wo.weight")
      # ffn weights
      for i in 0 ..< c.numLayers: f2.writeTensor(&"layers.{i}.ffn_norm.weight")
      for i in 0 ..< c.numLayers: f2.writeTensor(&"layers.{i}.feed_forward.w1.weight")
      for i in 0 ..< c.numLayers: f2.writeTensor(&"layers.{i}.feed_forward.w2.weight")
      for i in 0 ..< c.numLayers: f2.writeTensor(&"layers.{i}.feed_forward.w3.weight")

      f2.writeTensor("norm.weight")

      # write unused data
      # f2.writeTensor("freqs_cos")
      for i in 0 ..< c.seqLen * (c.dim div c.numHeads) div 2:
        f2.write(0.float32)
      # f2.writeTensor("freqs_sin")
      for i in 0 ..< c.seqLen * (c.dim div c.numHeads) div 2:
        f2.write(0.float32)

      f2.writeTensor("output.weight")

      f2.close()


loadAndExport("/media/me/ML/LLaMA/7B", "llama_7b_f32.bin")
