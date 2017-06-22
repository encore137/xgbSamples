package ru.megafon.smartReaction.utils

object Benchmarks {

  def simpleBenchmark(name: String)(f: => Unit): Unit = {
    val startTime = System.nanoTime()
    f
    val endTime = System.nanoTime()
    println(s"Execution time in $name: " + (endTime - startTime).toDouble / 1000000000 + " seconds" + "\n")
  }

}
