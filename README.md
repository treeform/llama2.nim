# Llama2 Implemented in Pure Nim with no dependencies.

This is a simple Nim port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), which is a stripped own simple implementation
to run inference of models with a [Llama](https://arxiv.org/pdf/2302.13971.pdf)-like transformer-based LLM architecture.

The code expects [`tokenizer.bin`](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) and [`stories15M.bin`](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) in the current directory.


```sh
>>> nim c -r -d:danger  src/llama2.nim -m:stories15M.bin -s:123 -i:"Mommy said "
```
```
build the Transformer via the model .bin file
build the Tokenizer via the tokenizer .bin file
build the Sampler
done loading
Mommy said
"Be careful, Sally. You don't want to hurt yourself."
Sally nodded and she started to count. She counted the apples, the cups, and the cookies. Then she quickly ran to the backyard. She saw a wide spot where she had grown a new cookie. She smiled and reached out to grab it.
But before she could grab the cookie, she heard a voice.
"Sally, what are you doing?"
Sally turned around and saw a big, mean-looking dog. She was scared, so she ran away.
When she got back to her house, she put the cookie back on the top of the square house. But she couldn't stop thinking about the dog. She got worried and started to count again.
Soon she came back to her bedroom and looked out of the window. She saw the dog in the living room. It was barking at her.
Sally thought it was a bad idea. She decided to call the dog back home and play with it instead. She ran outside and hid behind a tree.
When she got there, she saw the dog. She was too shy to say anything

achieved tok/s: 298.1149617785251
```

If you want to use Llama2 weights you first need to download them and use the converter script. Then Please follow instructions https://github.com/karpathy/llama2.c repo. To download them you need to agree to Meta's license agreement. I am working on a pure Nim converter that can read pytrouch's pickle files and generate safe tensors.
