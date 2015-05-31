package models

import optimizers.coordinate.BrentMethodOptimizer

import scala.Double._
import scala.math._

sealed class HingeLoss extends GenericLoss[RealFunction,DifferentiableRealFunction] (

  (y:Double) => (
    (x:Double) => max(0, 1 - x*y),
    (NegativeInfinity, PositiveInfinity)
    ),

  (y:Double) => (
    (a:Double) => y*a,
    (a:Double) => y,
    if (y == 1) (-1.0,0.0) else (0.0,1.0)
    )
)

sealed class LogisticLoss extends GenericLoss[RealFunction,DoublyDifferentiableRealFunction] (

  (y:Double) => (
    (x:Double) => log(1 + exp(-x*y)),
    (NegativeInfinity,PositiveInfinity)
    ),
  (y:Double) => (
    (a:Double) => if (a == 0 || y*a == -1) 0.0 else (1+a*y)*log(1+a*y) - a*y*log(-a*y),
    (a:Double) => y*log(1+a*y) - y*log(-a*y),
    (a:Double) => y/(1+y*a) + log(1+a*y) - y/a - log(-a*y),
    if (y == 1) (-1.0,0.0) else (0.0,1.0)
    )
)

sealed class RidgeLoss extends GenericLoss[RealFunction,DifferentiableRealFunction] (

  (y:Double) => (
    (x:Double) => (x - y) * (x - y),
    (NegativeInfinity,PositiveInfinity)
    ),
  (y:Double) => (
    (a:Double) => a*a/4 + a*y,
    (a:Double) => a/2 + y,
    (NegativeInfinity,PositiveInfinity)
    )
)