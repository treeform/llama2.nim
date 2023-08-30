import llama2

let
  checkpointPath = "stories15M.bin"
  tokenizerPath = "tokenizer.bin"
  temperature = 1.0
  pValue = 0.9
  seed = 123.uint64
  prompt = "Mommy said "
  steps = 256

echo "build the Transformer via the model .bin file"
var transformer = newTransformer(checkpointPath)

echo "build the Tokenizer via the tokenizer .bin file"
var tokenizer = newTokenizer(tokenizerPath, transformer.config.vocabSize)

echo "build the Sampler"
var sampler = newSampler(transformer.config.vocabSize, temperature, pValue, seed)

echo generate(
  transformer,
  tokenizer,
  sampler,
  prompt,
  steps.int32
) == "Mommy said \n\"Be careful, Sally. You don't want to hurt yourself.\"\nSally nodded and she started to count. She counted the apples, the cups, and the cookies. Then she quickly ran to the backyard. She saw a wide spot where she had grown a new cookie. She smiled and reached out to grab it.\nBut before she could grab the cookie, she heard a voice.\n\"Sally, what are you doing?\"\nSally turned around and saw a big, mean-looking dog. She was scared, so she ran away.\nWhen she got back to her house, she put the cookie back on the top of the square house. But she couldn't stop thinking about the dog. She got worried and started to count again.\nSoon she came back to her bedroom and looked out of the window. She saw the dog in the living room. It was barking at her.\nSally thought it was a bad idea. She decided to call the dog back home and play with it instead. She ran outside and hid behind a tree.\nWhen she got there, she saw the dog. She was too shy to say anything\n"
