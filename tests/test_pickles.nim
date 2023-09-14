import ../tools/pickles, json, flatty/hexprint

#    0: \x80 PROTO      4
#    2: K    BININT1    123
#    4: .    STOP
#highest protocol among opcodes = 2
#
doAssert %*123 == "\x80\x04K{.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      9
#   11: \x8c SHORT_BINUNICODE 'hello'
#   18: \x94 MEMOIZE    (as 0)
#   19: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*"hello" == "\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00\x8c\x05hello\x94.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      11
#   11: ]    EMPTY_LIST
#   12: \x94 MEMOIZE    (as 0)
#   13: (    MARK
#   14: K        BININT1    1
#   16: K        BININT1    2
#   18: K        BININT1    3
#   20: e        APPENDS    (MARK at 13)
#   21: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*[1, 2, 3] == "\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00]\x94(K\x01K\x02K\x03e.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      22
#   11: ]    EMPTY_LIST
#   12: \x94 MEMOIZE    (as 0)
#   13: (    MARK
#   14: K        BININT1    1
#   16: \x8c     SHORT_BINUNICODE 'two'
#   21: \x94     MEMOIZE    (as 1)
#   22: G        BINFLOAT   3.14
#   31: e        APPENDS    (MARK at 13)
#   32: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*[1, "two", 3.14] == "\x80\x04\x95\x16\x00\x00\x00\x00\x00\x00\x00]\x94(K\x01\x8c\x03two\x94G@\t\x1e\xb8Q\xeb\x85\x1fe.".pickleToJson()

#    0: \x80 PROTO      4
#    2: K    BININT1    8
#    4: .    STOP
#highest protocol among opcodes = 2
#
doAssert %*8 == "\x80\x04K\x08.".pickleToJson()

#    0: \x80 PROTO      4
#    2: K    BININT1    16
#    4: .    STOP
#highest protocol among opcodes = 2
#
doAssert %*16 == "\x80\x04K\x10.".pickleToJson()

#    0: \x80 PROTO      4
#    2: K    BININT1    32
#    4: .    STOP
#highest protocol among opcodes = 2
#
doAssert %*32 == "\x80\x04K .".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      11
#   11: \x8a LONG1      9223372036854775807
#   21: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*9223372036854775807 == "\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8a\x08\xff\xff\xff\xff\xff\xff\xff\x7f.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      11
#   11: \x8a LONG1      -9223372036854775808
#   21: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*(-9223372036854775808) == "\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8a\x08\x00\x00\x00\x00\x00\x00\x00\x80.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      10
#   11: G    BINFLOAT   3.14
#   20: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*3.14 == "\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G@\t\x1e\xb8Q\xeb\x85\x1f.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      10
#   11: G    BINFLOAT   1.0
#   20: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*1.0 == "\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      10
#   11: G    BINFLOAT   0.0
#   20: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*0.0 == "\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      10
#   11: G    BINFLOAT   -3.14
#   20: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*(-3.14) == "\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G\xc0\t\x1e\xb8Q\xeb\x85\x1f.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      22
#   11: ]    EMPTY_LIST
#   12: \x94 MEMOIZE    (as 0)
#   13: (    MARK
#   14: K        BININT1    1
#   16: \x8c     SHORT_BINUNICODE 'two'
#   21: \x94     MEMOIZE    (as 1)
#   22: G        BINFLOAT   3.14
#   31: e        APPENDS    (MARK at 13)
#   32: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*[1, "two", 3.14] == "\x80\x04\x95\x16\x00\x00\x00\x00\x00\x00\x00]\x94(K\x01\x8c\x03two\x94G@\t\x1e\xb8Q\xeb\x85\x1fe.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      27
#   11: }    EMPTY_DICT
#   12: \x94 MEMOIZE    (as 0)
#   13: (    MARK
#   14: \x8c     SHORT_BINUNICODE 'key'
#   19: \x94     MEMOIZE    (as 1)
#   20: \x8c     SHORT_BINUNICODE 'value'
#   27: \x94     MEMOIZE    (as 2)
#   28: \x8c     SHORT_BINUNICODE 'age'
#   33: \x94     MEMOIZE    (as 3)
#   34: K        BININT1    25
#   36: u        SETITEMS   (MARK at 13)
#   37: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*{"key": "value", "age": 25} == "\x80\x04\x95\x1b\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x03key\x94\x8c\x05value\x94\x8c\x03age\x94K\x19u.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      41
#   11: \x8c SHORT_BINUNICODE '__main__'
#   21: \x94 MEMOIZE    (as 0)
#   22: \x8c SHORT_BINUNICODE 'Pair'
#   28: \x94 MEMOIZE    (as 1)
#   29: \x93 STACK_GLOBAL
#   30: \x94 MEMOIZE    (as 2)
#   31: )    EMPTY_TUPLE
#   32: \x81 NEWOBJ
#   33: \x94 MEMOIZE    (as 3)
#   34: }    EMPTY_DICT
#   35: \x94 MEMOIZE    (as 4)
#   36: (    MARK
#   37: \x8c     SHORT_BINUNICODE 'a'
#   40: \x94     MEMOIZE    (as 5)
#   41: K        BININT1    1
#   43: \x8c     SHORT_BINUNICODE 'b'
#   46: \x94     MEMOIZE    (as 6)
#   47: K        BININT1    2
#   49: u        SETITEMS   (MARK at 36)
#   50: b    BUILD
#   51: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*{"build":"Pair.__main__","args":[],"kargs":{"a":1,"b":2}} == "\x80\x04\x95)\x00\x00\x00\x00\x00\x00\x00\x8c\x08__main__\x94\x8c\x04Pair\x94\x93\x94)\x81\x94}\x94(\x8c\x01a\x94K\x01\x8c\x01b\x94K\x02ub.".pickleToJson()

#    0: \x80 PROTO      4
#    2: \x95 FRAME      30
#   11: }    EMPTY_DICT
#   12: \x94 MEMOIZE    (as 0)
#   13: (    MARK
#   14: \x8c     SHORT_BINUNICODE 'key1'
#   20: \x94     MEMOIZE    (as 1)
#   21: \x8c     SHORT_BINUNICODE 'value1'
#   29: \x94     MEMOIZE    (as 2)
#   30: \x8c     SHORT_BINUNICODE 'key2'
#   36: \x94     MEMOIZE    (as 3)
#   37: h        BINGET     2
#   39: u        SETITEMS   (MARK at 13)
#   40: .    STOP
#highest protocol among opcodes = 4
#
doAssert %*{"key1": "value1", "key2": "value1"} == "\x80\x04\x95\x1e\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x04key1\x94\x8c\x06value1\x94\x8c\x04key2\x94h\x02u.".pickleToJson()
