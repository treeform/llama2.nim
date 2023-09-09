import std/json

type
  TensorStorage = object
    dataType*: string
    unknown1*: string
    device*: string
    unknown2*: int

  Tensor = object
    name*: string
    storage*: TensorStorage
    storageOffset*: int
    size*: seq[int]
    stride*: seq[int]
    requiresGrad*: bool

proc toTensors*(data: JsonNode): seq[Tensor] =

  for key, value in data:
    # _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks)
    if value["build"].getStr == "_rebuild_tensor_v2.torch._utils":
      var tensor = Tensor()
      tensor.name = key
      tensor.storage.dataType = value["args"][0][1].getStr
      tensor.storage.unknown1 = value["args"][0][2].getStr
      tensor.storage.device = value["args"][0][3].getStr
      tensor.storage.unknown2 = value["args"][0][1].getInt
      tensor.storageOffset = value["args"][1].getInt
      for v in value["args"][2]:
        tensor.size.add(v.getInt)
      for v in value["args"][3]:
        tensor.stride.add(v.getInt)
      tensor.requires_grad = value["args"][5].getBool

      result.add(tensor)
    else:
      quit("unknown object" & value["build"].getStr)
