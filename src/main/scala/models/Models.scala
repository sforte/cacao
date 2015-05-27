package models

import breeze.linalg.Vector
import models.RealFunction.{XFunction, Domain}

object RealFunction {
  type XFunction = Double => Double
  type Domain = (Double,Double)
}

class RealFunction(
  val func: XFunction,
  val domain: Domain)
  extends Serializable {
  def apply(x: Double) = func(x)
}

class DifferentiableRealFunction(
  override val func: XFunction,
  val derivative: RealFunction,
  override val domain: Domain)
  extends RealFunction(func,domain)

class DoublyDifferentiableRealFunction(
  override val func: XFunction,
  override val derivative: DifferentiableRealFunction,
  override val domain: Domain)
  extends DifferentiableRealFunction(func, derivative, domain)


trait LabelledRealFunction[+T<:RealFunction] extends Serializable {
  def apply(y: Double): T
}

object LabelledRealFunction {

  implicit def convert(arg: (Double => (XFunction, Domain))):
    LabelledRealFunction[RealFunction] =

    new LabelledRealFunction[RealFunction] {
      def apply(y: Double) = new RealFunction(arg(y)._1, arg(y)._2)
    }

  implicit def convert(arg: (Double => (XFunction, XFunction, Domain)))
    (implicit d: DummyImplicit):
     LabelledRealFunction[DifferentiableRealFunction] =

    new LabelledRealFunction[DifferentiableRealFunction] {
      def apply(y: Double) =
        new DifferentiableRealFunction(
          arg(y)._1, new RealFunction(arg(y)._2, arg(y)._3), arg(y)._3)
    }

  implicit def convert(arg: (Double => (XFunction, XFunction, XFunction, Domain)))
    (implicit d: DummyImplicit, d2:DummyImplicit):

    LabelledRealFunction[DoublyDifferentiableRealFunction] =

    new LabelledRealFunction[DoublyDifferentiableRealFunction] {
      def apply(y: Double) =
        new DoublyDifferentiableRealFunction(
          arg(y)._1, new DifferentiableRealFunction(arg(y)._2,
            new RealFunction(arg(y)._3, arg(y)._4), arg(y)._4), arg(y)._4)
    }
}

/*
  A function that maps a label y and a value x to a loss value.
  Used both to represent a primal and a dual loss.
 */
trait Loss[+P<:RealFunction,+D<:RealFunction] extends (Double => P) with Serializable {
  def conjugate: Loss[D,P]
}

case class GenericLoss[+P<:RealFunction,+D<:RealFunction]
  (primal: LabelledRealFunction[P], dual: LabelledRealFunction[D])
    extends Loss[P,D] {
  def conjugate = new GenericLoss(dual, primal)
  def apply(y: Double) = primal(y)
}

trait Regularizer extends Serializable {
  def primal(w: Vector[Double]): Double
  def dualGradient(w: Vector[Double]): Vector[Double]
  def lambda: Double

  def dual(v: Vector[Double]) = {
    val w = dualGradient(v)
    (w dot v) - primal(w)
  }
}

trait Model[L<:Loss[_,_]] {
  def n: Long
  def lambda: Double
  def loss: L
  def regularizer: Regularizer
}