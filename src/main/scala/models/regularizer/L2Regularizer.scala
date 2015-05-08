package models.regularizer

import breeze.linalg.Vector
import vectors.VectorOps.l2norm
import models.Regularizer

/**
 * Created by simone on 08/05/15.
 */
class L2Regularizer(val lambda: Double) extends Regularizer {

  def primal(w: Vector[Double]) = l2norm(w) / 2
  def dualGradient(w: Vector[Double]) = w

  def dual2(v: Vector[Double]) = l2norm(v)/2
}
