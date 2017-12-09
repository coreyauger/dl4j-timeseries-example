package io.surfkit.console

import java.io.File
import java.util.UUID
import java.util.concurrent.atomic.AtomicInteger

import org.joda.time.format.DateTimeFormat
import akka.actor.ActorSystem
import io.surfkit.ml.ML
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.joda.time.DateTime
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.indexing.NDArrayIndex

import scala.collection.Iterator
import scala.concurrent.Await
import scala.concurrent.duration._

object Main extends App {
  val HOST = "192.168.200.251"
  //val HOST = "localhost"
  val token = UUID.fromString("b475126c-b528-4539-a380-10f128a41d38")
  val user = UUID.fromString("f3adf9e4-5928-45c6-80c3-254fd2356a8d")

  val formatter = DateTimeFormat.forPattern("MM/dd/yyyy HH:mm")
  //val questrade = new QuestradeApi(true)

  @inline def defined(line: String) = {
    line != null && line.nonEmpty
  }
  Iterator.continually(scala.io.StdIn.readLine).takeWhile(defined(_)).foreach{line =>
    println("read " + line)
    CommandParser.parse(line.split(' ')).map { cmd =>
      cmd.mode match {
        case CommandParser.Mode.train =>
          ML.trainRnnTimeSeries
        case x => println(s"Unknown command '${x}'.")
      }
    }
  }

}

