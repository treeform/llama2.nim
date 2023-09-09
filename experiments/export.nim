import ../tools/pickles, ../tools/pth, std/streams, std/json, std/os,
    flatty/hexprint, std/strutils

proc loadAndExport(path: string, output: string) =
  echo readFile(path & "/params.json").parseJson()

  for i in 0 ..< 10:
    let pthPath = path & "/consolidated.0" & $i & ".pth"
    if fileExists(pthPath):
      echo pthPath

      let f = newFileStream(pthPath)

      let header = f.readStr(0x40)
      echo header.hexPrint()

      if header[0 ..< 4] != "PK\x03\x04":
        quit("Invalid magic bytes")

      let pkl = f.readStr(0x20000)
      let (data, bytes) = pkl.pickleToJson(false)

      echo bytes.toHex()

      echo hexPrint(pkl)

      # f.setPosition(bytes)
      # echo hexPrint(f.readStr(0x200))

      for tensor in toTensors(data):
        if tensor.name == "tok_embeddings.weight":
          echo tensor
        #echo tensor

loadAndExport("/media/me/ML/LLaMA/7B", "llama_7b_f32.bin")
