package distopt.utils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

// Dense Classification Point
case class ClassificationPoint(val label: Double, val features: Array[Double])

// Sparse Classifcation Point
case class SparseClassificationPoint(val index: Int, val label: Double, val features: SparseVector)

// Dense Regression Point
case class RegressionPoint(val label: Double, val features: Array[Double])

// Sparse Regression Point
case class SparseRegressionPoint(val label: Double, val features: SparseVector)

// Sparse Vector Implementation
class SparseVector(val indices: Array[Int], val values: Array[Double]) extends Serializable {

  def *=(c: Double) = values.transform(_ * c)

  def copy = new SparseVector(indices.clone, values.clone)

  def dot(o: SparseVector) = {

    var (prod,i,j) = (0.0,0,0)

    while (i < indices.size && j < o.indices.size) {
      if (indices(i) == o.indices(j)) prod += values(i) * o.values(j)
      if (indices(i) <= o.indices(j)) i += 1
      if (i < indices.size && indices(i) >= o.indices(j)) j += 1
    }

    prod
  }

  def dot(o: Array[Double]) = {

    var prod = 0.0

    for (i <- 0 until indices.size)
      prod += values(i) * o(indices(i))

    prod
  }

  def times(c: Double) = { val cp = copy; cp *= c; cp}

  def plus(dense: Array[Double]) : Array[Double] = {
    this.indices.zipWithIndex.foreach{ case(idx,i) => (dense(idx) = dense(idx)+this.values(i))}
    return dense
  }
}

class DoubleArray(arr: Array[Double]) {

  def += (o: SparseVector) {
    for (i <- 0 until o.indices.size)
      arr(o.indices(i)) += o.values(i)
  }

  def plus(plusArr: Array[Double]) : Array[Double] = {
    val retArr = (0 to plusArr.length-1).map( i => this.arr(i) + plusArr(i)).toArray
    return retArr
  }
  def times(c: Double) : Array[Double] = {
    val retArr = this.arr.map(x => x*c)
    return retArr
  }
}

object Implicits{
  implicit def arraytoDoubleArray(arr: Array[Double]) = new DoubleArray(arr)
}