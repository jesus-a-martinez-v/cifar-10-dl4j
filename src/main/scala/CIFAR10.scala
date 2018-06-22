import java.io.File

import org.apache.commons.io.FilenameUtils
import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.util.Try


object CIFAR10 extends App {

  // CIFAR dataset parameters.
  private val numberOfLabels = CifarLoader.NUM_LABELS
  private val numberOfSamples = CifarLoader.NUM_TRAIN_IMAGES
  private val numberOfTestSamples = CifarLoader.NUM_TEST_IMAGES
  private val dataPath = FilenameUtils.concat(System.getProperty("user.dir"), "/")

  val biasLearningRate = 2 * Configuration.learningRate
  DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)

  val model = getModel
  model.setListeners(new ScoreIterationListener(Configuration.printStatisticsFrequency))

  val trainingDataSet = new CifarDataSetIterator(Configuration.batchSize, numberOfSamples, Array(Configuration.height, Configuration.width, Configuration.channels), Configuration.preProcessCifar, true)
  val testDataSet = new CifarDataSetIterator(Configuration.batchSize, numberOfTestSamples, Array(Configuration.height, Configuration.width, Configuration.channels), Configuration.preProcessCifar, false)

  trainModel(model)
  evaluateModel(model)
  saveModel(model, "cifar_10_net.json")

  private def getModel = {
    val firstConvolutionLayer = new ConvolutionLayer
      .Builder(Array(4, 4), Array(1, 1), Array(0, 0))
      .name("cnn1")
      .convolutionMode(ConvolutionMode.Same)
      .nIn(3)
      .nOut(64)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU).learningRate(Configuration.learningRate)
      .biasInit(1e-2)
      .biasLearningRate(biasLearningRate)
      .build()

    val secondConvolutionLayer = new ConvolutionLayer
      .Builder(Array(4, 4), Array(1, 1), Array(0, 0))
      .name("cnn2")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(64)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(Configuration.learningRate)
      .biasInit(1e-2)
      .biasLearningRate(biasLearningRate)
      .build()

    val firstMaxPoolLayer = new SubsamplingLayer
      .Builder(PoolingType.MAX, Array(2, 2))
      .name("maxpool2")
      .build()

    val thirdConvolutionLayer = new ConvolutionLayer
      .Builder(Array(4, 4), Array(1, 1), Array(0, 0))
      .name("cnn3")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(96)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(Configuration.learningRate)
      .biasInit(1e-2)
      .biasLearningRate(biasLearningRate)
      .build()

    val fourthConvolutionLayer = new ConvolutionLayer.
      Builder(Array(4, 4), Array(1, 1), Array(0, 0))
      .name("cnn4")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(96)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(Configuration.learningRate)
      .biasInit(1e-2)
      .biasLearningRate(biasLearningRate)
      .build()

    val fifthConvolutionLayer = new ConvolutionLayer
      .Builder(Array(3, 3), Array(1, 1), Array(0, 0))
      .name("cnn5")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(128)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(Configuration.learningRate)
      .biasInit(1e-2)
      .biasLearningRate(biasLearningRate)
      .build()

    val sixthConvolutionLayer = new ConvolutionLayer
      .Builder(Array(3, 3), Array(1, 1), Array(0, 0))
      .name("cnn6")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(128)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(Configuration.learningRate)
      .biasInit(1e-2)
      .biasLearningRate(biasLearningRate)
      .build()

    val seventhConvolutionLayer = new ConvolutionLayer
      .Builder(Array(2, 2), Array(1, 1), Array(0, 0))
      .name("cnn7")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(256)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(Configuration.learningRate)
      .biasInit(1e-2)
      .biasLearningRate(biasLearningRate)
      .build()

    val eighthConvolutionLayer = new ConvolutionLayer
      .Builder(Array(2, 2), Array(1, 1), Array(0, 0))
      .name("cnn8")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(256)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(Configuration.learningRate)
      .biasInit(1e-2)
      .biasLearningRate(biasLearningRate)
      .build()

    val secondMaxPoolLayer = new SubsamplingLayer
      .Builder(PoolingType.MAX, Array(2, 2))
      .name("maxpool18")
      .build()

    val firstFullyConnectedLayer = new DenseLayer
      .Builder()
      .name("ffn1")
      .nOut(1024)
      .learningRate(1e-3)
      .biasInit(1e-3)
      .biasLearningRate(1e-3 * 2)
      .build()

    val firstDropOutLayer = new DropoutLayer
      .Builder()
      .name("dropout1")
      .dropOut(Configuration.dropOut)
      .build()

    val secondFullyConnectedLayer = new DenseLayer
      .Builder()
      .name("ffn2")
      .nOut(1024)
      .learningRate(Configuration.learningRate)
      .biasInit(1e-2)
      .biasLearningRate(biasLearningRate)
      .build()

    val secondDropOutLayer = new DropoutLayer
      .Builder()
      .name("dropout2")
      .dropOut(Configuration.dropOut)
      .build()

    val outputLayer = new OutputLayer
      .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .name("output")
      .nOut(numberOfLabels)
      .activation(Activation.SOFTMAX)
      .build()

    val layers = List(
      firstConvolutionLayer,
      secondConvolutionLayer,
      firstMaxPoolLayer,
      thirdConvolutionLayer,
      fourthConvolutionLayer,
      fifthConvolutionLayer,
      sixthConvolutionLayer,
      seventhConvolutionLayer,
      eighthConvolutionLayer,
      secondMaxPoolLayer,
      firstFullyConnectedLayer,
      firstDropOutLayer,
      secondFullyConnectedLayer,
      secondDropOutLayer,
      outputLayer)

    val configuration =
      new NeuralNetConfiguration.Builder()
        .seed(Configuration.randomSeed)
        .cacheMode(CacheMode.DEVICE)
        .updater(Updater.ADAM)
        .iterations(Configuration.iterations)
        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .regularization(true)
        .l1(1e-4)
        .l2(5 * 1e-4)
        .list(layers:_*)
        .backprop(true)
        .pretrain(false)
        .setInputType(InputType.convolutional(Configuration.height, Configuration.width, Configuration.channels))
        .build()

    val model = new MultiLayerNetwork(configuration)
    model.init()

    model
  }

  private def trainModel(model: MultiLayerNetwork): Unit =
    for (epoch <- 1 to Configuration.numberOfEpochs) {
      println(s"Epoch --------- $epoch")
      model.fit(trainingDataSet)
    }

  private def evaluateModel(model: MultiLayerNetwork): Unit = {
    val evaluation = new Evaluation(testDataSet.getLabels)
    while (testDataSet.hasNext) {
      val testBatch = testDataSet.next(Configuration.batchSize)
      val output = model.output(testBatch.getFeatureMatrix)

      evaluation.eval(testBatch.getLabels, output)
    }

    println(evaluation.stats())
  }

  private def saveModel(model: MultiLayerNetwork, fileName: String) = {
    val locationModelFile = new File(dataPath + fileName)

    if (Try(ModelSerializer.writeModel(model, locationModelFile, false)).isFailure) {
      println("Couldn't save model successfully.")
    }

    model
  }
}