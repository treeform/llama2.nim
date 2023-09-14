# PyTorch Model format reader.

import std/json, zippy/ziparchives

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

    memPointer*: pointer
    storageSize*: int

  PyTorchFile* = ref object
    zips: seq[ZipArchiveReader]
    tensors*: seq[Tensor]

proc toTensors*(data: JsonNode): seq[Tensor] =

  for key, value in data:
    # _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks)
    if value["build"].getStr == "_rebuild_tensor_v2.torch._utils":
      var tensor = Tensor()
      tensor.name = key
      echo value["args"]
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
        tensor.storageSize = 1
        for s in tensor.size:
          tensor.storageSize *= s

      result.add(tensor)
    else:
      quit("unknown object" & value["build"].getStr)
