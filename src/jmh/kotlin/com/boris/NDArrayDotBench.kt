package com.boris

import NDArrayDot
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray
import io.kinference.utils.runBlocking
import kotlinx.coroutines.Dispatchers
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import toNDArray
import java.util.*

// Run with `jmh` task
@Suppress("unused")
@State(Scope.Benchmark)
open class NDArrayDotBench {
    private lateinit var a: FloatNDArray
    private lateinit var b: FloatNDArray
    private lateinit var c: MutableFloatNDArray

    @Setup(Level.Iteration)
    fun init() {
//        val (a, b) = generateRandom(1024, 1024, 1024, seed = 42)

        // slowest, may take 50 minutes, but is best in showing the speed-up of the new algo
//        val (a, b) = generateRandom(4096, 4096, 4096, seed = 42)

//        val (a, b) = generateRandom(1024, 64, 1024 * 16, seed = 42)
        val (a, b) = generateRandom(2048, 64, 2048*32, seed = 42)
//        val (a, b) = generateRandom(1024, 16, 1024 * 64, seed = 42)

        // worst impact of array copying, bcz blockSize of the source is bad & old algorithm is also fast here
//        val (a, b) = generateRandom(1024 - 1, 16 - 1, 1024 * 64 - 1, seed = 42)

        // this case shows that the overhead of copying is considerably small even in the worst possible case
//        val (a, b) = generateRandom(1024 - 1, 16 * 16 - 1, 1024 * 64 - 1, seed = 42)

        // this case also shows dramatic improvement
//        val (a, b) = generateRandom(1024, 4096, 256, seed = 42)
//        val (a, b) = generateRandom(2048, 2048*4, 2048/4, seed = 42)
//        val (a, b) = generateRandom(1024 * 64, 256, 64, seed = 42)
//        val (a, b) = generateRandom(4096 * 64, 256, 64, seed = 42)
//        val (a, b) = generateRandom(64, 1024 * 16, 64, seed = 42)
//        val (a, b) = generateRandom(64, 1024 * 32, 16, seed = 42)
//        val (a, b) = generateRandom(64, 1024 * 32, 16, seed = 42)
//        val (a, b) = generateRandom(64, 1024 * 128, 4, seed = 42)
//        val (a, b) = generateRandom(64, 1024 * 256, 2, seed = 42)
//        val (a, b) = generateRandom(64, 1024 * 256, 1, seed = 42)
//        val (a, b) = generateRandom(1024 * 2, 1024 / 4, 64, seed = 42)
//        val (a, b) = generateRandom(1024 * 8, 256, 16, seed = 42)
//        val (a, b) = generateRandom(1024 * 32, 256, 4, seed = 42)
//        val (a, b) = generateRandom(1024 * 64, 256, 2, seed = 42)
//        val (a, b) = generateRandom(1024 * 4, 128, 64, seed = 42)
//        val (a, b) = generateRandom(1024 * 16, 128, 16, seed = 42)
//        val (a, b) = generateRandom(1024 * 64, 128, 4, seed = 42)
//        val (a, b) = generateRandom(1024 * 128, 128, 2, seed = 42)
//        val (a, b) = generateRandom(64, 64, 64, seed = 42)
//        val (a, b) = generateRandom(16, 256, 16, seed = 42)
//        val (a, b) = generateRandom(4, 256 * 8, 4, seed = 42)
//        val (a, b) = generateRandom(2, 256 * 16, 2, seed = 42)
        this.a = a.toNDArray()
        this.b = b.toNDArray()
        this.c = Array(a.size) { FloatArray(b[0].size) }.toNDArray()
    }

//    @Benchmark
    fun bench_old(bh: Blackhole) = runBlocking(Dispatchers.Default) {
        bh.consume(NDArrayDot.old(a, b, c))
    }
//    @Benchmark
//    fun bench_new(bh: Blackhole) = runBlocking(Dispatchers.Default) {
//        bh.consume(NDArrayDot.new(a, b, c))
//    }
//    @Benchmark
//    fun bench_copy(bh: Blackhole) = runBlocking(Dispatchers.Default) {
//        bh.consume(NDArrayDot.copy(a, b, c))
//    }
//    @Benchmark
    fun bench_resize(bh: Blackhole) = runBlocking(Dispatchers.Default) {
        bh.consume(NDArrayDot.resize(a, b, c))
    }


    @Benchmark
    fun bench_resize_parallel(bh: Blackhole) = runBlocking(Dispatchers.Default) {
        bh.consume(NDArrayDot.resize_parallel(a, b, c))
    }
    @Benchmark
    fun bench_resize_parallel_lesslaunches(bh: Blackhole) = runBlocking(Dispatchers.Default) {
        bh.consume(NDArrayDot.resize_parallel_lesslaunches(a, b, c))
    }
//    @Benchmark
    fun bench_old_parallel(bh: Blackhole) = runBlocking(Dispatchers.Default) {
        bh.consume(NDArrayDot.old_parallel(a, b, c))
    }

    @Benchmark
    fun bench_cupertank(bh: Blackhole) = runBlocking(Dispatchers.Default) {
        bh.consume(NDArrayDot.cupertankParallel(a, b, c))
    }

    @Suppress("SameParameterValue")
    private fun generateRandom(
        m: Int,
        t: Int,
        n: Int,
        seed: Long = Random().nextLong()
    ): Pair<Array<FloatArray>, Array<FloatArray>> {
        val random = Random(seed)

//        fun randomArray(m: Int, n: Int): Array<FloatArray> {
//            return random.Floats()
//                .limit(m.toLong() * n)
//                .asSequence()
//                .chunked(n)
//                .map { it.toFloatArray() }
//                .toList()
//                .toTypedArray()
//        }

        fun randomArray(m: Int, n: Int): Array<FloatArray> {
            return Array(m) {
                random.doubles().limit(n.toLong()).toArray().map { it.toFloat() }.toFloatArray()
            }
        }

        val a = randomArray(m, t)
        val b = randomArray(t, n)

        return a to b
    }
}
