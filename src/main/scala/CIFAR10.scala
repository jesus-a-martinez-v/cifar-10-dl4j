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
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.util.Try


object CIFAR10 extends App {
  private val height = 32
  private val width = 32
  private val channels = 3
  private val numberOfLabels = CifarLoader.NUM_LABELS
  private val numberOfSamples = CifarLoader.NUM_TRAIN_IMAGES
  private val numberOfTestSamples = CifarLoader.NUM_TEST_IMAGES
  private val batchSize = 100
  private val iterations = 1
  private val freIterations = 50
  private val randomSeed = 42
  private val preProcessCifar = false
  private val numberOfEpochs = 50
  private val dataPath = FilenameUtils.concat(System.getProperty("user.dir"), "/")

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)

  val model = trainModelByCifarWithNet()
  // val uiServer = UIServer.getInstance()
  val statsStorage = new InMemoryStatsStorage

  // uiServer.attach(statsStorage)

  model.setListeners(
    new StatsListener(statsStorage),
    new ScoreIterationListener(freIterations))

  val cifar = new CifarDataSetIterator(batchSize, numberOfSamples, Array(height, width, channels), preProcessCifar, true)
  val cifarEval = new CifarDataSetIterator(batchSize, numberOfTestSamples, Array(height, width, channels), preProcessCifar, false)

  for (epoch <- 1 to numberOfEpochs) {
    println(s"Epoch --------- $epoch")
    model.fit(cifar)
  }

  val evaluation = new Evaluation(cifarEval.getLabels)
  while (cifarEval.hasNext) {
    val testDataSet = cifarEval.next(batchSize)
    val output = model.output(testDataSet.getFeatureMatrix)

    evaluation.eval(testDataSet.getLabels, output)
  }

  println(evaluation.stats())

  saveModel(model, "trainModelByCifarWithAlexNet_model.json")

  private def trainModelByCifarWithNet() = {
    val firstConvolutionLayer = new ConvolutionLayer
      .Builder(Array(4, 4), Array(1, 1), Array(0, 0))
      .name("cnn1")
      .convolutionMode(ConvolutionMode.Same)
      .nIn(3)
      .nOut(64)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU).learningRate(1e-2)
      .biasInit(1e-2)
      .biasLearningRate(1e-2 * 2)
      .build()

    val secondConvolutionLayer = new ConvolutionLayer
      .Builder(Array(4, 4), Array(1, 1), Array(0, 0))
      .name("cnn2")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(64)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(1e-2)
      .biasInit(1e-2)
      .biasLearningRate(1e-2 * 2)
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
      .learningRate(1e-2)
      .biasInit(1e-2)
      .biasLearningRate(1e-2 * 2)
      .build()

    val fourthConvolutionLayer = new ConvolutionLayer.
      Builder(Array(4, 4), Array(1, 1), Array(0, 0))
      .name("cnn4")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(96)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(1e-2)
      .biasInit(1e-2)
      .biasLearningRate(1e-2 * 2)
      .build()

    val fifthConvolutionLayer = new ConvolutionLayer
      .Builder(Array(3, 3), Array(1, 1), Array(0, 0))
      .name("cnn5")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(128)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(1e-2)
      .biasInit(1e-2)
      .biasLearningRate(1e-2 * 2)
      .build()

    val sixthConvolutionLayer = new ConvolutionLayer
      .Builder(Array(3, 3), Array(1, 1), Array(0, 0))
      .name("cnn6")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(128)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(1e-2)
      .biasInit(1e-2)
      .biasLearningRate(1e-2 * 2)
      .build()

    val seventhConvolutionLayer = new ConvolutionLayer
      .Builder(Array(2, 2), Array(1, 1), Array(0, 0))
      .name("cnn7")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(256)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(1e-2)
      .biasInit(1e-2)
      .biasLearningRate(1e-2 * 2)
      .build()

    val eighthConvolutionLayer = new ConvolutionLayer
      .Builder(Array(2, 2), Array(1, 1), Array(0, 0))
      .name("cnn8")
      .convolutionMode(ConvolutionMode.Same)
      .nOut(256)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .activation(Activation.RELU)
      .learningRate(1e-2)
      .biasInit(1e-2)
      .biasLearningRate(1e-2 * 2)
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
      .dropOut(0.2)
      .build()

    val secondFullyConnectedLayer = new DenseLayer
      .Builder()
      .name("ffn2")
      .nOut(1024)
      .learningRate(1e-2)
      .biasInit(1e-2)
      .biasLearningRate(1e-2 * 2)
      .build()

    val secondDropOutLayer = new DropoutLayer
      .Builder()
      .name("dropout2")
      .dropOut(0.2)
      .build()

    val outputLayer = new OutputLayer
      .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .name("output")
      .nOut(numberOfLabels)
      .activation(Activation.SOFTMAX)
      .build()

    val configuration = new NeuralNetConfiguration.Builder()
      .seed(randomSeed)
      .cacheMode(CacheMode.DEVICE)
      .updater(Updater.ADAM)
      .iterations(iterations)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .l1(1e-4)
      .regularization(true)
      .l2(5 * 1e-4)
      .list()
      .layer(0, firstConvolutionLayer)
      .layer(1, secondConvolutionLayer)
      .layer(2, firstMaxPoolLayer)
      .layer(3, thirdConvolutionLayer)
      .layer(4, fourthConvolutionLayer)
      .layer(5, fifthConvolutionLayer)
      .layer(6, sixthConvolutionLayer)
      .layer(7, seventhConvolutionLayer)
      .layer(8, eighthConvolutionLayer)
      .layer(9, secondMaxPoolLayer)
      .layer(10, firstFullyConnectedLayer)
      .layer(11, firstDropOutLayer)
      .layer(12, secondFullyConnectedLayer)
      .layer(13, secondDropOutLayer)
      .layer(14, outputLayer)
      .backprop(true)
      .pretrain(false)
      .setInputType(InputType.convolutional(height, width, channels))
      .build()

    val model = new MultiLayerNetwork(configuration)
    model.init()

    model
  }

  private def saveModel(model: MultiLayerNetwork, fileName: String) = {
    val locationModelFile = new File(dataPath + fileName)

    if (Try(ModelSerializer.writeModel(model, locationModelFile, false)).isFailure) {
      println("Saving model wasn't successful")
    }

    model
  }
}