package models.loss

import models._
import Double.{NegativeInfinity,PositiveInfinity}
import scala.math.{exp, log}

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