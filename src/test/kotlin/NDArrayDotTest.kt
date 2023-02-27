import io.kinference.ndarray.arrays.DoubleNDArray
import io.kinference.ndarray.arrays.MutableDoubleNDArray
import java.util.*
import kotlin.streams.asSequence
import kotlin.streams.asStream
import kotlin.test.Test
import kotlin.test.assertEquals

class NDArrayDotTest {
    @Test
    fun test_old() = test(NDArrayDot::old)
    @Test
    fun test_new() = test(NDArrayDot::new)
    @Test
    fun test_copy() = test(NDArrayDot::copy)
    @Test
    fun test_resize() = test(NDArrayDot::resize)

    private fun test(matmul: (DoubleNDArray, DoubleNDArray, MutableDoubleNDArray) -> Unit) {
        for ((caseName, a, b, expected) in cases) {
            val c = zeroesOfShape(expected.shape[0], expected.shape[1])
            matmul(a, b, c)
            assertDeepEquals(expected.toDoubleArray(), c.toDoubleArray(), case = caseName)
        }
    }

    private fun zeroesOfShape(m: Int, n: Int): MutableDoubleNDArray {
        return (0 until m)
            .map { (0 until n).map { 0.0 }.toDoubleArray() }
            .toTypedArray().toNDArray().copyIfNotMutable()
    }

    private fun assertDeepEquals(
        expected: Array<DoubleArray>,
        c: Array<DoubleArray>,
        tolerance: Double = 1e-6,
        case: String
    ) {
        assertEquals(expected.size, c.size)
        assert(expected.isNotEmpty())
        for (i in expected.indices) {
            assertEquals(expected[i].size, c[i].size, "Sizes of row $i do not match, case '$case'")
            assert(expected[i].isNotEmpty())
            for (j in expected[0].indices) {
                assertEquals(expected[i][j], c[i][j], tolerance, "Elements at ($i, $j) do not match, case '$case'")
            }
        }
    }

    private data class Case(
        val name: String,
        val a: DoubleNDArray,
        val b: DoubleNDArray,
        val expected: DoubleNDArray,
    ) {
        init {
            require(a.shape[0] == expected.shape[0])
            require(a.linearSize > 0)

            require(a.shape[1] == b.shape[0])
            require(b.linearSize > 0)

            require(b.shape[1] == expected.shape[1])
        }
    }

    private val case_3x3x3_hardcoded = Case(
        "3x3 x 3x3 hardcoded",
        a = arrayOf(
            doubleArrayOf(0.5, 1.1, 2.3),
            doubleArrayOf(2.2, 3.3, 4.5),
            doubleArrayOf(1.5, 4.9, 3.5),
        ).toNDArray(),
        b = arrayOf(
            doubleArrayOf(1.7, 3.2, 8.3),
            doubleArrayOf(8.2, 9.2, 1.3),
            doubleArrayOf(2.4, 6.2, 0.3),
        ).toNDArray(),
        expected = arrayOf(
            doubleArrayOf(15.39, 25.98, 6.27),
            doubleArrayOf(41.6, 65.3, 23.9),
            doubleArrayOf(51.13, 71.58, 19.87),
        ).toNDArray()
    )

    private val case_123x597x314_generated: Case = generateRandom(
        123, 597, 314,
        seed = 7, name = "123x597 597x314 generated"
    )

    private val case_1024x1024x1024_generated: Case = generateRandom(
        1024, 1024, 1024,
        seed = 42, name = "1024x1024 1024x1024 generated"
    )

    private val cases = listOf(
        case_3x3x3_hardcoded,
        case_123x597x314_generated,
        case_1024x1024x1024_generated,
    )

    private fun generateRandom(m: Int, t: Int, n: Int, seed: Long = Random().nextLong(), name: String): Case {
        val random = Random(seed)

        fun randomArray(m: Int, n: Int): Array<DoubleArray> {
            return random.doubles()
                .limit(m.toLong() * n)
                .asSequence()
                .chunked(n)
                .map { it.toDoubleArray() }
                .toList()
                .toTypedArray()
        }

        val a = randomArray(m, t)
        val b = randomArray(t, n)

        val expected = (0 until m)
            .asSequence()
            .map { i -> (0 until n).map { i to it } }
            .asStream().parallel()
            .map { row ->
                row.map { (i, j) -> (0 until t).sumOf { a[i][it] * b[it][j] } }
                    .toDoubleArray()
            }
            .toList().toTypedArray()

        return Case(name, a.toNDArray(), b.toNDArray(), expected.toNDArray())
    }
}