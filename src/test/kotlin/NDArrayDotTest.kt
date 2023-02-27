import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray
import io.kinference.ndarray.toFloatArray
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

    private fun test(matmul: (FloatNDArray, FloatNDArray, MutableFloatNDArray) -> Unit) {
        for ((caseName, a, b, expected) in cases) {
            val c = zeroesOfShape(expected.shape[0], expected.shape[1])
            matmul(a, b, c)
            assertDeepEquals(expected.toFloatArray(), c.toFloatArray(), case = caseName)
        }
    }

    private fun zeroesOfShape(m: Int, n: Int): MutableFloatNDArray {
        return (0 until m)
            .map { (0 until n).map { 0.0 }.toFloatArray() }
            .toTypedArray().toNDArray().copyIfNotMutable()
    }

    private fun assertDeepEquals(
        expected: Array<FloatArray>,
        c: Array<FloatArray>,
        tolerance: Float = 1e-3f,
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
        val a: FloatNDArray,
        val b: FloatNDArray,
        val expected: FloatNDArray,
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
            floatArrayOf(0.5f, 1.1f, 2.3f),
            floatArrayOf(2.2f, 3.3f, 4.5f),
            floatArrayOf(1.5f, 4.9f, 3.5f),
        ).toNDArray(),
        b = arrayOf(
            floatArrayOf(1.7f, 3.2f, 8.3f),
            floatArrayOf(8.2f, 9.2f, 1.3f),
            floatArrayOf(2.4f, 6.2f, 0.3f),
        ).toNDArray(),
        expected = arrayOf(
            floatArrayOf(15.39f, 25.98f, 6.27f),
            floatArrayOf(41.6f, 65.3f, 23.9f),
            floatArrayOf(51.13f, 71.58f, 19.87f),
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

        fun randomArray(m: Int, n: Int): Array<FloatArray> {
            return random.doubles()
                .limit(m.toLong() * n)
                .asSequence()
                .chunked(n)
                .map { it.toFloatArray() }
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
                row.map { (i, j) -> (0 until t).sumOf { a[i][it] * b[it][j].toDouble() } }
                    .toFloatArray()
            }
            .toList().toTypedArray()

        return Case(name, a.toNDArray(), b.toNDArray(), expected.toNDArray())
    }
}