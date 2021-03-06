import sbt.Keys._

import sbt._

val publishVersion = "0.0.1-SNAPSHOT"

name := """dl4j-timeseries-example"""

scalaVersion := "2.11.11"

organization in ThisBuild := "io.surfkit"

version := publishVersion

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

scalacOptions in (Compile,doc) := Seq()

resolvers in ThisBuild += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

resolvers in ThisBuild += "Apache Snapshots" at "https://repository.apache.org/snapshots/"

lazy val `dl4j-timeseries-example` =
  (project in file("."))
    .settings(commonLibSettings:_*)

val dl4jV = "0.9.1"

resolvers += Resolver.mavenLocal

//classpathTypes += "maven-plugin"

lazy val commonLibSettings = Seq(
  name := s"derpy-hooves-rnn",
  libraryDependencies ++= Seq(
    "com.github.scopt"             %% "scopt"                              % "3.5.0",
    "org.deeplearning4j"            % "deeplearning4j-core"                % dl4jV,
    "org.deeplearning4j"           %% "deeplearning4j-ui"                  % dl4jV,
    "org.nd4j"                      % "nd4j-native-platform"               % dl4jV
  )
)

isSnapshot in ThisBuild := true

