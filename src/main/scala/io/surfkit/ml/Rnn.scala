package io.surfkit.ml

import java.io.File
import java.net.URL
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.util.{Collections, Random, UUID}

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.{FileSplit, NumberedFileInputSplit}
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import scala.io.Source

/**
  * Created by suroot on 30/11/17.
  */
object ML {

  //'baseDir': Base directory for the data. Change this if you want to save the
  //data somewhere else
  val baseDir = new File("src/main/resources/uci/")
  val baseTrainDir = new File(baseDir, "train")
  val featuresDirTrain = new File(baseTrainDir, "features")
  val labelsDirTrain = new File(baseTrainDir, "labels")
  val baseTestDir = new File(baseDir, "test")
  val featuresDirTest = new File(baseTestDir, "features")
  val labelsDirTest = new File(baseTestDir, "labels")


  def trainRnnTimeSeries = {
    if (!baseDir.exists())
      downloadUCIData()

    // ----- Load the training data -----
    //Note that we have 450 training files for features: train/features/0.csv
    //through train/features/449.csv
    val trainFeatures = new CSVSequenceRecordReader()
    trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449))
    val trainLabels = new CSVSequenceRecordReader()
    trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449))

    val miniBatchSize = 10
    val numLabelClasses = 6
    val trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)

    //Normalize the training data
    val normalizer = new NormalizerStandardize()
    normalizer.fit(trainData)              //Collect training data statistics
    trainData.reset()

    //Use previously collected statistics to normalize on-the-fly. Each DataSet
    //returned by 'trainData' iterator will be normalized
    trainData.setPreProcessor(normalizer)
    // ----- Load the test data -----
    //Same process as for the training data.
    val testFeatures = new CSVSequenceRecordReader();
    testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 148))
    val testLabels = new CSVSequenceRecordReader()
    testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 148))

    val testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)

    testData.setPreProcessor(normalizer)   //Note that we are using the exact
    //same normalization process as the
    //training data


    // ----- Configure the network -----
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123)    //Random number generator seed for
      //improved repeatability. Optional.
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .learningRate(0.005)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps
      //with this data set
      .gradientNormalizationThreshold(0.5)
      .list()
      .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build())
      .pretrain(false).backprop(true).build()

    val net = new MultiLayerNetwork(conf)
    net.init()

    net.setListeners(new ScoreIterationListener(20))   //Print the score (loss
    //function value) every 20 iterations


    // -- Train the network, evaluating the test set performance at each epoch --
    val nEpochs = 40
    (0 to nEpochs).foreach{ i =>
      net.fit(trainData)

      //Evaluate on the test set:
      val evaluation = net.evaluate(testData)
      println(evaluation.stats())
      println(evaluation.confusionToString())

      testData.reset()
      trainData.reset()
    }

    println("----- Example Complete -----")
  }


  //This method downloads the data, and converts the "one time series per line"
  //format into a suitable
  //CSV sequence format that DataVec (CsvSequenceRecordReader) and DL4J can read.
  def downloadUCIData() = {
    val url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data"


    val html = Source.fromURL(url)
    val data = html.mkString

    val lines = data.split ("\n")

    //Create directories
    baseDir.mkdir ()
    baseTrainDir.mkdir ()
    featuresDirTrain.mkdir ()
    labelsDirTrain.mkdir ()
    baseTestDir.mkdir ()
    featuresDirTest.mkdir ()
    labelsDirTest.mkdir ()

    val contentAndLabels = lines.zipWithIndex.map{ case (line,ind) =>
      (line.replaceAll (" +", "\n"), ind / 100)
    }

    //Randomize and do a train/test split:
    scala.util.Random.shuffle(contentAndLabels.toList)

    val nTrain = 450 //75% train, 25% test
    var trainCount = -1
    var testCount = -1
    contentAndLabels.foreach{  p =>
      //Write output in a format we can read, in the appropriate locations
      val (outPathFeatures, outPathLabels) =
      if (trainCount < nTrain) {
        trainCount = trainCount+1
        (new File (featuresDirTrain, trainCount + ".csv"), new File (labelsDirTrain, trainCount + ".csv"))
      } else {
        testCount = testCount+1
        (new File (featuresDirTest, testCount + ".csv"), new File (labelsDirTest, testCount + ".csv"))
      }
      Files.write(Paths.get(outPathFeatures.getAbsolutePath), p._1 .getBytes(StandardCharsets.UTF_8))
      Files.write(Paths.get(outPathLabels.getAbsolutePath), p._2.toString.getBytes(StandardCharsets.UTF_8))
    }
  }

}
