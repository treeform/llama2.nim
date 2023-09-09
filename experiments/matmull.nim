## This file contains several matmul experiments



proc matMul(xout: ptr float32, x: ptr float32, w: ptr float32, n: cint, d: cint) =
  ## 4 CPU lane mat mull (Compilers might be able to SIMD this)

  for i in 0 ..< d:

    var
      val0 = 0.0'f32
      val1 = 0.0'f32
      val2 = 0.0'f32
      val3 = 0.0'f32

    var j = 0
    let offset = i * n
    while j < n:
      val0 += w[offset + j+0] * x[j+0]
      val1 += w[offset + j+1] * x[j+1]
      val2 += w[offset + j+2] * x[j+2]
      val3 += w[offset + j+3] * x[j+3]
      j += 4
    while j < n:
      val0 += w[offset + j] * x[j]
      inc j

    xout[i] = val0 + val1 + val2 + val3



proc matMul(xout: ptr float32, x: ptr float32, w: ptr float32, n: cint, d: cint) =
  # rotation matMul it does d x n or n x d depending on who is larger

  if d <= n:
    for i in 0 ..< d:
      var val = 0.0'f32
      for j in 0 ..< n:
        val += w[i * n + j] * x[j]
      xout[i] = val
  else:
    for i in 0 ..< d:
      xout[i] = 0

    for j in 0 ..< n:
      for i in 0 ..< d:
        xout[i] = xout[i] + w[i * n + j] * x[j]


import nimsimd/avx2

{.push header: "immintrin.h".}
func mm_fmadd_ps*(a, b, c: M128): M128 {.importc: "_mm_fmadd_ps".}
func mm256_fmadd_ps*(a, b, c: M256): M256 {.importc: "_mm256_fmadd_ps".}
{.pop.}


proc matMul(xout: ptr float32, x: ptr float32, w: ptr float32, n: cint, d: cint) =
  ## 8 lane simd

  let
    x = cast[ptr UncheckedArray[float32]](x)
    w = cast[ptr UncheckedArray[float32]](w)

  for i in 0 ..< d:

    var
      val: M256
      val0: float32
      j = 0
    let offset = i * n
    while j < n:
      let
        wv = mm256_loadu_ps(w[offset + j].addr)
        xv = mm256_loadu_ps(x[j].addr)
      val = mm256_fmadd_ps(wv, xv, val)
      j += 8

    while j < n:
      val0 += w[offset + j] * x[j]
      inc j

    var tmp = cast[array[8, float32]](val)
    for i in 0 ..< 8:
      val0 += tmp[i]
    xout[i] = val0




## Creates threads when it needs too:

const numThreads = 12
type Work = object
  dest, w, x: ptr[float32]
  n, start, stop: int
var threads: array[0..numThreads, Thread[Work]]

proc threadFunc(work: Work) {.thread.} =
  var
    dest = work.dest
    w = work.w
    x = work.x
    n = work.n
    start = work.start
    stop = work.stop

  for i in start ..< stop:
    var val = 0.0'f32
    for j in 0 ..< n:
      val += w[i * n + j] * x[j]
    dest[i] = val

proc matMul(dest: ptr float32, x: ptr float32, w: ptr float32, n: int32, d: int32) =
  ## Matrix vector multiply.
  # W (d,n) @ x (n,) -> xout (d,)
  # by far the most amount of time is spent inside this little function

  let perThread = d div numThreads

  for t in 0 ..< numThreads:
    let start = t * perThread
    let stop = if t < numThreads - 1:
      (t + 1) * perThread
    else:
      d

    createThread(threads[t], threadFunc, Work(
      dest: dest,
      w: w,
      x: x,
      n: n,
      start: start,
      stop: stop
    ))

  joinThreads(threads)






# Using 12 threads that are working

import std/locks

const numThreads = 12
type Job = object
  run: bool
  dest, w, x: ptr[float32]
  n, start, stop: int

var jobs: array[0..numThreads, Job]
var jobLocks: array[0..numThreads, Lock]
var threads: array[0..numThreads, Thread[int]]

proc threadFunc(t: int) {.thread.} =
  while true:
    withLock(jobLocks[t]):
      if jobs[t].run:
        var
          dest = jobs[t].dest
          w = jobs[t].w
          x = jobs[t].x
          n = jobs[t].n
          start = jobs[t].start
          stop = jobs[t].stop

        for i in start ..< stop:
          var val = 0.0'f32
          for j in 0 ..< n:
            val += w[i * n + j] * x[j]
          dest[i] = val

        jobs[t].run = false

for t in 0 ..< numThreads:
  initLock(jobLocks[t])
  createThread(threads[t], threadFunc, t)

proc matMul(dest: ptr float32, x: ptr float32, w: ptr float32, n: int32, d: int32) =
  ## Matrix vector multiply.

  let perThread = d div numThreads

  for t in 0 ..< numThreads:
    let start = t * perThread
    let stop = if t == numThreads - 1:
        d.int
      else:
        (t + 1) * perThread.int

    withLock(jobLocks[t]):
      jobs[t] = Job(
        run: true,
        dest: dest,
        w: w,
        x: x,
        n: n,
        start: start,
        stop: stop
      )

  while true:
    var done = true
    for t in 0 ..< numThreads:
      withLock(jobLocks[t]):
        if jobs[t].run:
          done = false
        else:
          discard
    if done:
      break




# Using 12 threads that are waiting on efficiently wait
# whe not in use

import std/locks

const numThreads = 12
type Job = object
  run: bool
  dest, w, x: ptr[float32]
  n, start, stop: int

var jobs: array[0..numThreads, Job]
var jobLocks: array[0..numThreads, Lock]
var jobConds: array[0..numThreads, Cond]
var threads: array[0..numThreads, Thread[int]]

proc threadFunc(t: int) {.thread.} =
  while true:
    withLock(jobLocks[t]):
      if jobs[t].run:
        var
          dest = jobs[t].dest
          w = jobs[t].w
          x = jobs[t].x
          n = jobs[t].n
          start = jobs[t].start
          stop = jobs[t].stop

        for i in start ..< stop:
          var val = 0.0'f32
          for j in 0 ..< n:
            val += w[i * n + j] * x[j]
          dest[i] = val

        jobs[t].run = false
      else:
        jobConds[t].wait(jobLocks[t])

for t in 0 ..< numThreads:
  initLock(jobLocks[t])
  initCond(jobConds[t])
  createThread(threads[t], threadFunc, t)

proc matMul(dest: ptr float32, x: ptr float32, w: ptr float32, n: int32, d: int32) =
  ## Matrix vector multiply.

  let perThread = d div numThreads

  for t in 0 ..< numThreads:
    let start = t * perThread
    let stop = if t == numThreads - 1:
        d.int
      else:
        (t + 1) * perThread.int

    withLock(jobLocks[t]):
      jobs[t] = Job(
        run: true,
        dest: dest,
        w: w,
        x: x,
        n: n,
        start: start,
        stop: stop
      )
      jobConds[t].signal()

  while true:
    var done = true
    for t in 0 ..< numThreads:
      withLock(jobLocks[t]):
        if jobs[t].run:
          done = false
        else:
          discard
    if done:
      break
