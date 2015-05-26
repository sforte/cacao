package test

import breeze.numerics.abs
import models.{DifferentiableRealFunction, GenericLoss, RealFunction}
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.optimization.GoalType
import org.apache.commons.math.optimization.univariate.BrentOptimizer

import scala.Double._
import scala.math._

object ConjugateTest extends App {

//  val loss = new QModifiedL1Loss(1,1,0.1)
//
//  testConjugate(loss(0), loss.conjugate(0))

  def testConjugate(function: RealFunction, conj: RealFunction, step: Double = 0.1, tol: Double = 10E-1) = {
    (conj.domain._1 to conj.domain._2 by step).foreach( x => {
      println(x, conjugate(function,x), conj(x), abs(conjugate(function, x) - conj(x)))
//      assert(abs(conjugate(function, x) - conj(x)) <= tol)
    })
  }

  def conjugate(function: RealFunction, x: Double) = {
    class Func(x: Double) extends UnivariateRealFunction {
      def value(a: Double) = x*a - function(a)
    }

    val brent = new BrentOptimizer
    brent.setMaximalIterationCount(10000)
    brent.setRelativeAccuracy(0)
    brent.setAbsoluteAccuracy(4*Math.ulp(1d))

    val func = new Func(x)
    val a = brent.optimize(func, GoalType.MAXIMIZE, -1000, 1000, 0)

    func.value(a)
  }
}

