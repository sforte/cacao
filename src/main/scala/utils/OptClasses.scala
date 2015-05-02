package distopt.utils

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector => Vec}

import breeze.linalg.{SparseVector => SV}
import scala.collection.mutable.ArrayBuffer

/*
  Common optimized vector operations
 */
object VectorOps {

  def dot(a: Vec, b: Vec): Double = {
    (a,b) match {
      case (a: SparseVector, b: SparseVector) => dot(a,b)
      case (a: SparseVector, b: DenseVector) => dot(a,b)
      case (a: DenseVector, b: SparseVector) => dot(b,a)
      case (a: DenseVector, b: DenseVector) => dot(a,b)
    }
  }

  def dot(a: SparseVector, b: SparseVector) : Double = {
    require(a.size == b.size)

    var (prod,i,j) = (0.0,0,0)

    while (i < a.indices.size && j < b.indices.size) {
      if (a.indices(i) == b.indices(j)) prod += a.values(i) * b.values(j)
      if (a.indices(i) <= b.indices(j)) i += 1
      if (i < a.indices.size && a.indices(i) >= b.indices(j)) j += 1
    }

    prod
  }

  def dot(a: SparseVector, b: DenseVector) : Double = {
    require(a.size == b.size)

    var prod = 0.0

    for (i <- 0 until a.indices.size)
      prod += a.values(i) * b.values(a.indices(i))

    prod
  }

  def dot(a: DenseVector, b: DenseVector) : Double = {
    require(a.size == b.size)

    var prod = 0.0

    for (i <- 0 until a.size)
      prod += a.values(i) * b.values(i)

    prod
  }

  def times(a: Vec, c: Double) : Vec = {
    a match {
      case (a: SparseVector) => times(a,c)
      case (a: DenseVector) => times(a,c)
    }
  }

  def times(a: SparseVector, c: Double) : SparseVector = {
    val res = a.copy
    res.values.transform(_ * c)
    res
  }

  def times(a: DenseVector, c: Double) : DenseVector = {
    val res = a.copy
    res.values.transform(_ * c)
    res
  }

  def plus(a: Vec, b: DenseVector) : DenseVector = {
    a match {
      case (a: SparseVector) => plus(a,b)
      case (a: DenseVector) => plus(a,b)
    }
  }

  def plus(a: DenseVector, b: Vec) : DenseVector = plus(b,a)

  def plus(a: Vec, b: Vec) : Vec = {
    (a,b) match {
      case (a: SparseVector, b: SparseVector) => plus(a,b)
      case (a: SparseVector, b: DenseVector) => plus(a,b)
      case (a: DenseVector, b: SparseVector) => plus(b,a)
      case (a: DenseVector, b: DenseVector) => plus(a,b)
    }
  }

  def plus(a: SparseVector, b: SparseVector) : SparseVector = {

    require(a.size == b.size)

    val (indices,values) = (ArrayBuffer[Int](),ArrayBuffer[Double]())
    var (i,j) = (0,0)

    while (i < a.indices.size || j < b.indices.size) {
      if (j == b.indices.size || (i < a.indices.size && a.indices(i) < b.indices(j))) {
        indices += a.indices(i)
        values += a.values(i)
        i += 1
      } else if (i == a.indices.size || a.indices(i) > b.indices(j)) {
        indices += b.indices(j)
        values += b.values(j)
        j += 1
      } else {
        indices += a.indices(i)
        values += a.values(i) + b.values(j)
        i += 1
        j += 1
      }
    }

    new SparseVector(a.size, indices.toArray, values.toArray)
  }

  def minus(a: Vec, b: Vec) = plus(a, times(b,-1))

  def plus(a: SparseVector, b: DenseVector) : DenseVector = {
    require(a.size == b.size)

    val res = b.copy

    for (i <- 0 until a.indices.size)
      res.values(a.indices(i)) += a.values(i)

    res
  }

  def plus(a: DenseVector, b: DenseVector) : DenseVector = {
    require(a.size == b.size)

    val res = b.copy

    for (i <- 0 until a.size)
      res.values(i) += a.values(i)

    res
  }

  def plusEqual(a: DenseVector, b: Vec, c: Double) : DenseVector = {
    b match {
      case (b: SparseVector) => plusEqual(a,b,c)
      case (b: DenseVector) => plusEqual(a,b,c)
    }
  }

  def plusEqual(a: DenseVector, b: DenseVector, c: Double) : DenseVector = {
    require(a.size == b.size)

    for (i <- 0 until b.size)
      a.values(i) += b.values(i) * c
    a
  }

  def plusEqual(a: DenseVector, b: SparseVector, c: Double) : DenseVector = {
    require(a.size == b.size)

    for (i <- 0 until b.indices.size)
      a.values(b.indices(i)) += b.values(i) * c
    a
  }

  def l2norm(w: Vec) = dot(w,w)
  def l1norm(w: Vec) = w match {
    case (w: SparseVector) => w.values.map(math.abs).sum
    case (w: DenseVector) => w.values.map(math.abs).sum
  }
}