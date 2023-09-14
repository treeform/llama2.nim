## Python Pickle format reader

import json, std/strformat, flatty/binny, std/strutils, std/tables

proc pr(c: char): string =
  ## Python-style-repr for characters.
  ## returns character if it's a normal visible ASCII, or prints \x80 instead.
  let asciiVal = cast[uint8](c)
  if asciiVal >= 33 and asciiVal <= 126:
    return $c  # $c converts the char to its string representation
  else:
    return "\\x" & toHex(asciiVal, 2).toLowerAscii()

func swap*(v: float64): float64 {.inline.} =
  let tmp = cast[array[2, uint32]](v)
  let i = (swap(tmp[0]).uint64 shl 32) or swap(tmp[1])
  return cast[float64](i)

proc pop(arr: var JsonNode): JsonNode =
  result = arr[arr.len - 1]
  var newArr = newJArray()
  for i in 0 ..< arr.len - 1:
    newArr.add(arr[i])
  arr = newArr

proc pickleToJsonSize*(bin: string, interactive = true): (JsonNode, int) =

  var
    indent = ""
    marks: seq[int]
    numMemoize = 0
    stackGlobal: seq[(string, string)]
    memo: Table[int, JsonNode]
    stacks: seq[JsonNode]
    stack = newJArray()

  template display(what: string) =
    if interactive:
      echo &"{address:5d}: {c.pr:4} {indent}" & what

  # dis step
  var i = 0
  while i < bin.len:
    let c = bin[i]
    let address = i
    case c
      of '\x80':
        inc i
        let v = bin.readUInt8(i)
        inc i
        display(&"PROTO      {v}")
      of 'K':
        inc i
        let v = bin.readUInt8(i)
        inc i
        display(&"BININT1    {v}")
        stack.add %v
      of 'M':
        inc i
        let v = bin.readInt16(i)
        i += 2
        display(&"BININT2    {v}")
        stack.add %v
      of 'J':
        inc i
        let v = bin.readInt32(i)
        i += 4
        display(&"BININT     {v}")
        stack.add %v
      of '\x8a':
        inc i
        let numLen = bin.readUInt8(i).int
        inc i
        if numLen == 8:
          let v = bin.readInt64(i)
          i += 8
          display(&"LONG1    {v}")
          stack.add %v
        else:
          quit("unsupported LONG1")
      of '\x95':
        inc i
        let v = bin.readUInt64(i)
        i += 8
        display(&"FRAME    {v}")
      of '.':
        inc i
        display(&"STOP")
        break
      of '\x8c':
        inc i
        let strLen = bin.readUInt8(i).int
        inc i
        let v = bin[i ..< i + strLen]
        i += strLen
        display(&"SHORT_BINUNICODE '{v}'")
        stack.add %v
      of 'X':
        inc i
        let strLen = bin.readUInt32(i).int
        i += 4
        let v = bin[i ..< i + strLen]
        i += strLen
        display(&"BINUNICODE '{v}'")
        stack.add %v
      of '\x94':
        inc i
        display(&"MEMOIZE (as {numMemoize})")
        memo[numMemoize] = stack[^1]
        inc numMemoize
      of ']':
        inc i
        display(&"EMPTY_LIST")
        stack.add(newJArray())
      of ')':
        inc i
        display(&"EMPTY_TUPLE")
        stack.add(newJArray())
      of '}':
        inc i
        display(&"EMPTY_DICT")
        stack.add(newJObject())
      of '\x81':
        inc i
        display(&"NEWOBJ")
      of 'b':
        inc i
        display(&"BUILD")
        let kargs = stack.pop()
        let args = stack.pop()
        let name = stack.pop()
        let v = newJObject()
        v["build"] = name
        v["args"] = args
        v["kargs"] = kargs
        stack.add(v)
      of '(':
        marks.add(i)
        inc i
        display(&"MARK")
        indent.add("    ")
        stacks.add(stack)
        stack = newJArray()
      of 'e':
        inc i
        display(&"APPENDS (MARK at {marks[^1]})")
        discard marks.pop()
        indent.setLen(indent.len - 4)
        var v = stack
        stack = stacks.pop()
        if stack.pop().kind != JArray:
          quit("JArray expected")
        stack.add(v)
      of 'u':
        inc i
        display(&"SETITEMS   (MARK at {marks[^1]})")
        discard marks.pop()
        indent.setLen(indent.len - 4)
        var v = newJObject()
        for i in 0 ..< stack.len div 2:
          v[stack[i*2].getStr()] = stack[i*2+1]
        stack = stacks.pop()
        if stack.pop().kind != JObject:
          quit("JObject expected")
        stack.add(v)
      of 't':
        inc i
        display(&"TUPLE      (MARK at {marks[^1]})")
        discard marks.pop()
        indent.setLen(indent.len - 4)
        var v = stack
        stack = stacks.pop()
        # if stack.pop().kind != JArray:
        #   quit("JArray expected")
        stack.add(v)
      of '\x85':
        inc i
        display(&"TUPLE1")
        var v = newJArray()
        v.add(stack.pop())
        stack.add(v)
      of '\x86':
        inc i
        display(&"TUPLE2")
        var v = newJArray()
        v.add(stack.pop())
        v.add(stack.pop())
        stack.add(v)
      of 'G':
        inc i
        let v = bin.readFloat64(i).swap()
        i += 8
        display(&"BINFLOAT    {v}")
        stack.add %v
      of '\x93':
        inc i
        display(&"STACK_GLOBAL")
        let
          module = stack.pop().getStr
          name = stack.pop().getStr
        #stackGlobal.add(())
        let v = module & "." & name
        stack.add %v
      of 'c':
        inc i
        var name = ""
        while bin[i] != '\n':
          name.add bin[i]
          inc i
        inc i
        var module = ""
        while bin[i] != '\n':
          module.add bin[i]
          inc i
        inc i
        display(&"GLOBAL     '{name} {module}'")
        let v = module & "." & name
        stack.add %v
      of 'q':
        inc i
        let slot = bin.readUInt8(i).int
        inc i
        display(&"BINPUT     {slot}")
        memo[slot.int] = stack[^1]
      of 'r':
        inc i
        let slot = bin.readUInt32(i).int
        i += 4
        display(&"LONG_BINPUT {slot}")
        memo[slot.int] = stack[^1]

      of 'h':
        inc i
        let slot = bin.readUInt8(i).int
        inc i
        display(&"BINGET     {slot}")
        stack.add(memo[slot.int])

      of 'Q':
        inc i
        display(&"BINPERSID")
        ## ???  discard stack.pop()

      of '\x89':
        inc i
        display(&"NEWFALSE")
        stack.add(%false)

      of 'R':
        inc i
        display(&"REDUCE")
        #echo stack
        let args = stack.pop()
        let name = stack.pop()
        let v = newJObject()
        v["build"] = name
        v["args"] = args
        stack.add(v)
      else:
        quit("??: " & c.pr)

  if interactive:
    echo "highest protocol among opcodes = 2"
    # echo stackGlobal
    # echo stacks
    # echo stack
  if stack.len != 1:
    quit("Items left on the stack")
  return (stack[0], i)

proc pickleToJson*(bin: string, interactive = true): JsonNode =
  return pickleToJsonSize(bin, interactive)[0]
