name := "cifar-10-dl4j"

version := "0.1"

scalaVersion := "2.12.4"

classpathTypes += "maven-plugin"

libraryDependencies ++= {
  val dl4jVersion = "0.9.1"

  Seq(
    "org.nd4j" % "nd4j-native-platform" % dl4jVersion,
    "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
    "org.datavec" % "datavec-api" % dl4jVersion,
    "org.deeplearning4j" % "deeplearning4j-ui_2.11" % dl4jVersion,
    "org.datavec" % "datavec-data-image" % dl4jVersion)
}