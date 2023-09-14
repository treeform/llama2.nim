

# Using 12 threads that are waiting on efficiently wait
# whe not in use

import std/locks

proc `+`[T](p: ptr[T], n: SomeInteger): ptr[T] =
  cast[ptr[T]](cast[uint64](p) + n.uint64 * sizeof(T).uint64)

proc `[]`[T](p: ptr[T], n: SomeInteger): T =
  let p2 = p + n
  return p2[]

proc `[]=`[T](p: ptr[T], n: SomeInteger, v: T) =
  let p2 = p + n
  p2[] = v

proc `+`(p: pointer, n: SomeInteger): pointer =
  cast[pointer](cast[uint64](p) + n.uint64)

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

proc matMul*(dest: ptr float32, x: ptr float32, w: ptr float32, n: int32, d: int32) =
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
