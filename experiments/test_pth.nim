
import ../tools/pickles, ../tools/pth, std/streams, std/json

let f = newFileStream("/media/me/ML/LLaMA/7B/consolidated.00.pth")
let header = f.readStr(0x40)
#echo header.hexPrint()
let pkl = f.readStr(0x20000)
#echo pkl.hexPrint()
let data = pkl.pickleToJson()
echo pretty(data)

for tensor in toTensors(data):
  echo tensor
