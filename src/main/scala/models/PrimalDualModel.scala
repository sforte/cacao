package models

trait RealFunction extends ((Double,Double) => Double) with Serializable {
  def domain(y: Double) : (Double,Double)
}

trait PrimalDualModel {

  def primalLoss: RealFunction
  def dualLoss: RealFunction
  def lambda: Double
  def initAlpha(y: Double): Double
}