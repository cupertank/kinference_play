import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray

fun Array<FloatArray>.toNDArray(): MutableFloatNDArray {
    val result = MutableFloatNDArray(intArrayOf(this.size, this[0].size))
    copyBlocks(this, result.array.blocks)
    return result
//    return MutableFloatNDArray(intArrayOf(this.size, this[0].size)) {
//        this[it / this[0].size][it % this[0].size]
//    }
}

fun FloatNDArray.toFloatArray(): Array<FloatArray> {
    val (result, _) = emptyBlocks(shape, blockSize = shape[1])
    copyBlocks(array.blocks, result)
    return result
//    val blocksInRow = array.blocksNum / shape[0]
//    val blockSize = array.blockSize
//    return Array(shape[0]) { i ->
//        val row = FloatArray(shape[1])
//        for (block in i * blocksInRow until (i + 1) * blocksInRow) {
//            array.blocks[block].copyInto(row, (block - i * blocksInRow) * blockSize)
//        }
//        row
//    }
}

fun emptyBlocks(shape: IntArray, blockSize: Int): Pair<Array<FloatArray>, Int> {
    val blocksInRow: Int
    val lastBlockSize: Int

    if (shape[1] % blockSize == 0) {
        blocksInRow = shape[1] / blockSize
        lastBlockSize = blockSize
    } else {
        blocksInRow = shape[1] / blockSize + 1
        lastBlockSize = shape[1] % blockSize
    }

    val array = Array(shape[0] * blocksInRow) { blockI ->
        val actualBlockSize = if ((blockI + 1) % blocksInRow == 0) lastBlockSize else blockSize
        FloatArray(actualBlockSize)
    }
    return array to blocksInRow
}

fun copyBlocks(srcBlocks: Array<FloatArray>, dstBlocks: Array<FloatArray>) {
    if (srcBlocks === dstBlocks) {
        return
    }

    var srcBlockI = 0
    var srcBlockOffset = 0

    var dstBlockI = 0
    var dstBlockOffset = 0

    while (srcBlockI < srcBlocks.size && dstBlockI < dstBlocks.size) {
        val srcBlock = srcBlocks[srcBlockI]
        val dstBlock = dstBlocks[dstBlockI]

        val copiedSize = minOf(srcBlock.size - srcBlockOffset, dstBlock.size - dstBlockOffset)

        srcBlock.copyInto(
            dstBlock,
            destinationOffset = dstBlockOffset,
            startIndex = srcBlockOffset,
            endIndex = srcBlockOffset + copiedSize
        )

        srcBlockOffset += copiedSize
        if (srcBlockOffset == srcBlock.size) {
            srcBlockOffset = 0
            srcBlockI++
        }

        dstBlockOffset += copiedSize
        if (dstBlockOffset == dstBlock.size) {
            dstBlockOffset = 0
            dstBlockI++
        }
    }
}
