# PyTorch Model format reader.

import std/json, zippy_extra, std/os, pickles, std/strutils

type
  TensorStorage* = object
    dataType*: string
    fileName*: string
    device*: string
    unknown2*: int

  Tensor* = ref object
    name*: string
    storage*: TensorStorage
    storageOffset*: int
    size*: seq[int]
    stride*: seq[int]
    requiresGrad*: bool

    data*: pointer
    dataSize*: int

  TorchData* = ref object
    zips: seq[ZipArchiveReader]
    tensors*: seq[Tensor]

proc `+`*(p: pointer, n: SomeInteger): pointer =
  cast[pointer](cast[uint64](p) + n.uint64)

proc `[]`*(p: pointer, n: SomeInteger): uint8 =
  return cast[ptr[uint8]](p + n)[]

proc toTensors*(data: JsonNode): seq[Tensor] =

  for key, value in data:
    # _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks)
    if value["build"].getStr == "_rebuild_tensor_v2.torch._utils":
      var tensor = Tensor()
      tensor.name = key
      tensor.storage.dataType = value["args"][0][1].getStr
      tensor.storage.fileName = value["args"][0][2].getStr
      tensor.storage.device = value["args"][0][3].getStr
      tensor.storage.unknown2 = value["args"][0][1].getInt
      tensor.storageOffset = value["args"][1].getInt
      for v in value["args"][2]:
        tensor.size.add(v.getInt)
      for v in value["args"][3]:
        tensor.stride.add(v.getInt)
      tensor.requires_grad = value["args"][5].getBool

      if tensor.size.len > 0:
        tensor.dataSize = 1
        for s in tensor.size:
          tensor.dataSize *= s

      result.add(tensor)
    else:
      quit("unknown object" & value["build"].getStr)

proc loadTorchData*(path: string): TorchData =

  result = TorchData()

  for i in 0 ..< 100:
    let pthPath = path & "/consolidated.0" & $i & ".pth"
    if fileExists(pthPath):

      let reader = openZipArchive(pthPath)
      result.zips.add(reader)

      doAssert reader.extractFile("consolidated/version").strip() == "3"

      let
        dataPickle = reader.extractFile("consolidated/data.pkl")
        dataJson = dataPickle.pickleToJson(false)

      let tensors = toTensors(dataJson)

      for tensor in tensors:
        tensor.data = reader.getPointer("consolidated/data/" & tensor.storage.fileName)
        result.tensors.add(tensor)

proc find*(torchData: TorchData, name: string): Tensor =
  for tensor in torchData.tensors:
    if tensor.name == name:
      return tensor

proc close*(torchData: TorchData) =
  for zip in torchData.zips:
    zip.close()
