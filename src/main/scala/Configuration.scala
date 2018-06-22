import com.typesafe.config.ConfigFactory

object Configuration {
  private val configuration = ConfigFactory.load()
  private val imageConfig = configuration.getConfig("image")
  private val hyperParametersConfig = configuration.getConfig("hyperParameters")

  val height: Int = imageConfig.getInt("height")
  val width: Int = imageConfig.getInt("width")
  val channels: Int = imageConfig.getInt("channels")

  val batchSize: Int = hyperParametersConfig.getInt("batchSize")
  val iterations: Int = hyperParametersConfig.getInt("iterations")
  val randomSeed: Int = hyperParametersConfig.getInt("randomSeed")
  val numberOfEpochs: Int = hyperParametersConfig.getInt("numberOfEpochs")
  val learningRate: Double = hyperParametersConfig.getDouble("learningRate")
  val dropOut: Double = hyperParametersConfig.getDouble("dropOut")
  val preProcessCifar: Boolean = hyperParametersConfig.getBoolean("preProcessCifar")
  val printStatisticsFrequency: Int = hyperParametersConfig.getInt("printStatisticsFrequency")
}