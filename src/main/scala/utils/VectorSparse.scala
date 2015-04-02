package utils

import scala.collection.mutable.Map

object VectorSparse {
  implicit def mapToSparseVector(map : Map[Int,Float]) = new VectorSparse(map)
}

/*
The class represents a sparse array with various optimized
operations implemented.
 */
class VectorSparse(val vector : Map[Int,Float]) extends Iterable[Int] {

  var scalar = 1.0

  def copy = {val cl : VectorSparse = vector.clone; cl.scalar = scalar; cl}

  def apply(i : Int) = scalar * vector.getOrElse(i,0.0f)

  def update(i : Int, v : Double) = vector.update(i,(v/scalar).toFloat)

  override def size = vector.size

  override def iterator = vector.keysIterator

  def * (s : Double) = copy *= s

  def / (s : Double) = copy /= s

  def + (that : VectorSparse) = this.copy += that

  def - (that : VectorSparse) = this.copy -= that

  def * (that : VectorSparse) = if(this.size <= that.size) p(this,that) else p(that,this)

  def *= (s : Double) = {if(s == 0) vector.clear else scalar *= s; this}

  def /= (s : Double) = {scalar /= s; this}

  def += (that : VectorSparse) = {for(w <- that) this(w) = this(w) + that(w); this}

  def -= (that : VectorSparse) = {for(w <- that) this(w) = this(w) - that(w); this}

  def p(a : VectorSparse, b : VectorSparse) = {var sum = 0.0; a.foreach{w => sum += a(w) * b(w)}; sum}
}