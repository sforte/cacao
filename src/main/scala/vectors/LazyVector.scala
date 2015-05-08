package vectors

import breeze.linalg.{SparseVector, DenseVector, Vector}

abstract class LazyVector extends Vector[Double] with Serializable {
  // unsupported
  final override def update(i: Int, v: Double) = ???
}

class LazyMappedVector (vec: Vector[Double], func: (Double => Double)) extends LazyVector {

  override def length: Int = vec.length

  override def copy = new LazyMappedVector(vec, func)

  override def activeSize: Int = vec.activeSize

  override def apply(i: Int) = func(vec(i))

  override def activeIterator = activeKeysIterator zip activeValuesIterator

  override def activeKeysIterator = vec.activeKeysIterator

  override def activeValuesIterator = vec.activeValuesIterator.map(func)

  override def repr = this
}

class LazyScaledVector (vec: Vector[Double], s: Double) extends LazyMappedVector(vec, _*s)

class LazySumVector (a: DenseVector[Double], b: DenseVector[Double], s: Double) extends LazyVector {

  require(a.size == b.size)

  override def length = a.length

  override def copy = new LazySumVector(a,b,s)

  override def activeSize = a.activeSize

  override def apply(i: Int) = a.data(i) + b.data(i)*s

  override def activeIterator = activeKeysIterator zip activeValuesIterator

  override def activeKeysIterator = a.activeKeysIterator

  override def activeValuesIterator = activeKeysIterator.map(k => a(k) + b(k)*s)

  override def repr = this
}

